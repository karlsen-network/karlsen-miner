use futures::prelude::*;
use native_tls::TlsConnector as NativeTlsConnector;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_native_tls::TlsConnector;
use tokio_util::codec::Framed;

mod statum_codec;

use crate::client::stratum::statum_codec::{ErrorCode, MiningNotify, MiningSubmit, NewLineJsonCodecError, StratumLine};
use crate::client::stratum::statum_codec::{
    MiningSubscribe, SetExtranonce, StratumCommand, StratumError, StratumLinePayload, StratumResult,
};
use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::PartialBlock;
use crate::{miner::MinerManager, Error, Uint256};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use log::{error, info, warn};
use num::Float;
use rand::{rng, RngCore};
use statum_codec::NewLineJsonCodec;
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tokio::task;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};

//const DIFFICULTY_1_TARGET: Uint256 = Uint256([0x00000000ffff0000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000]);
const DIFFICULTY_1_TARGET: (u64, i16) = (0xffffu64, 208); // 0xffff 2^208
const LOG_RATE: Duration = Duration::from_secs(30);

type BlockHandle = JoinHandle<Result<(), PollSendError<StratumLine>>>;

#[derive(Default)]
pub struct ShareStats {
    pub accepted: AtomicU64,
    pub stale: AtomicU64,
    pub low_diff: AtomicU64,
    pub duplicate: AtomicU64,
    pub shares_pending: Mutex<HashMap<u32, String>>,
}

static mut SHARE_STATS: Option<Arc<ShareStats>> = None;

impl Display for ShareStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Shares: {}{}{}{}Pending: {}",
            match self.accepted.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Accepted: {} ", v),
            },
            match self.stale.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Stale: {} ", v),
            },
            match self.low_diff.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Low difficulty: {} ", v),
            },
            match self.duplicate.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Duplicate: {} ", v),
            },
            self.shares_pending.try_lock().unwrap().len()
        )
    }
}

#[allow(dead_code)]
pub struct StratumHandler {
    log_handler: JoinHandle<()>,

    //client: Framed<TcpStream, NewLineJsonCodec>,
    send_channel: Sender<StratumLine>,
    stream: Pin<Box<dyn Stream<Item = Result<StratumLine, NewLineJsonCodecError>>>>,
    miner_address: String,
    mine_when_not_synced: bool,
    devfund_address: Option<String>,
    devfund_percent: u16,
    mining_dev: Option<bool>,
    block_template_ctr: Arc<AtomicU16>,

    target_pool: Uint256,
    target_real: Uint256,
    nonce_mask: u64,
    nonce_fixed: u64,
    extranonce: Option<String>,
    last_stratum_id: Arc<AtomicU32>,

    shares_stats: Arc<ShareStats>,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,
}

#[async_trait(?Send)]
impl Client for StratumHandler {
    fn add_devfund(&mut self, address: String, percent: u16) {
        self.devfund_address = Some(address);
        self.devfund_percent = percent;
    }

    async fn register(&mut self) -> Result<(), Error> {
        let mut id = { Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst)) };
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Subscribe(
                    MiningSubscribe::MiningSubscribeDefault((
                        format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                        //self.extranonce.clone().unwrap_or("0xffffffff".into())
                    )),
                )),
                jsonrpc: None,
                error: None,
            })
            .await?;
        id = Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst));

        let pay_address = match &self.devfund_address {
            Some(devfund_address) if self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent => {
                self.mining_dev = Some(true);
                info!("Mining to devfund");
                devfund_address.clone()
            }
            _ => {
                self.mining_dev = Some(false);
                self.miner_address.clone()
            }
        };
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((
                    pay_address.clone(),
                    "x".into(),
                ))),
                jsonrpc: None,
                error: None,
            })
            .await?;
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        info!("Waiting for stuff");
        loop {
            {
                if self.devfund_percent > 0 {
                    let block_count = self.block_template_ctr.load(Ordering::SeqCst);
                    let mining_dev = self.mining_dev.unwrap_or(true);

                    match (mining_dev, block_count <= self.devfund_percent) {
                        (false, true) => return Ok(()), // not mining dev but under threshold
                        (true, false) => return Ok(()), // mining dev but over threshold
                        _ => {}                         // continue mining
                    }
                }
            }
            match self.stream.try_next().await? {
                Some(msg) => self.handle_message(msg, miner).await?,
                None => return Err("stratum message payload is empty".into()),
            }
        }
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl StratumHandler {
    pub async fn connect(
        address: String,
        miner_address: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        use_ssl: bool,
    ) -> Result<Box<Self>, Error> {
        info!("Connecting to {}", address);
        let socket = TcpStream::connect(&address).await?;

        let (send_channel, recv) = mpsc::channel::<StratumLine>(3);

        let stream: Pin<Box<dyn Stream<Item = Result<StratumLine, NewLineJsonCodecError>>>> = if use_ssl {
            info!("Using SSL connection");
            let connector = TlsConnector::from(
                NativeTlsConnector::builder()
                    .danger_accept_invalid_certs(true)
                    .danger_accept_invalid_hostnames(true)
                    .build()?,
            );
            let tls_stream = connector.connect("", socket).await?;
            let client = Framed::new(tls_stream, NewLineJsonCodec::new());
            let (sink, stream) = client.split();
            tokio::spawn(async move { ReceiverStream::new(recv).map(Ok).forward(sink).await });
            Box::pin(stream)
        } else {
            info!("Using TCP connection");
            let client = Framed::new(socket, NewLineJsonCodec::new());
            let (sink, stream) = client.split();
            tokio::spawn(async move { ReceiverStream::new(recv).map(Ok).forward(sink).await });
            Box::pin(stream)
        };

        let share_state = unsafe {
            #[allow(static_mut_refs)]
            if SHARE_STATS.is_none() {
                SHARE_STATS = Some(Arc::new(ShareStats::default()));
            }
            #[allow(static_mut_refs)]
            SHARE_STATS.clone().unwrap()
        };
        let last_stratum_id = Arc::new(AtomicU32::new(0));
        let (block_channel, block_handle) = Self::create_block_channel(
            send_channel.clone(),
            miner_address.clone(),
            last_stratum_id.clone(),
            share_state.clone(),
        );
        Ok(Box::new(Self {
            log_handler: task::spawn(Self::log_shares(share_state.clone())),
            stream,
            send_channel,
            miner_address,
            mine_when_not_synced,
            devfund_address: None,
            devfund_percent: 0,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((rng().next_u64() % 10_000u64) as u16))),
            target_pool: Default::default(),
            target_real: Default::default(),
            nonce_mask: 0,
            nonce_fixed: 0,
            extranonce: None,
            last_stratum_id,
            shares_stats: share_state,
            mining_dev: None,
            block_channel,
            block_handle,
        }))
    }

    fn create_block_channel(
        send_channel: Sender<StratumLine>,
        miner_address: String,
        last_stratum_id: Arc<AtomicU32>,
        share_stats: Arc<ShareStats>,
    ) -> (Sender<BlockSeed>, BlockHandle) {
        let (send, recv) = mpsc::channel::<BlockSeed>(1);

        let handle = tokio::spawn(async move {
            ReceiverStream::new(recv)
                .map(move |block_seed| {
                    let (nonce, id) = match block_seed {
                        BlockSeed::PartialBlock { ref nonce, ref id, .. } => (nonce, id),
                        BlockSeed::FullBlock(_) => unreachable!(),
                    };
                    let msg_id = last_stratum_id.fetch_add(1, Ordering::SeqCst);
                    {
                        share_stats.shares_pending.try_lock().unwrap().insert(
                            msg_id,
                            //SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
                            id.clone(), //block_seed.clone()
                        );
                    }
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitShort((
                                miner_address.clone(),
                                id.into(),
                                format!("{:016x}", nonce),
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                })
                .map(Ok)
                .forward(PollSender::new(send_channel))
                .await
        });
        (send, handle)
    }

    async fn handle_message(&mut self, msg: StratumLine, miner: &mut MinerManager) -> Result<(), Error> {
        match msg.clone() {
            StratumLine { id, payload, error: None, .. } => {
                match payload {
                    StratumLinePayload::StratumResult { result } if id.is_some() => {
                        match result {
                            StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                                if let Some(_jobid) = self
                                    .shares_stats
                                    .shares_pending
                                    .try_lock()
                                    .unwrap()
                                    .remove(&id.expect("We checked id is not none"))
                                {
                                    self.shares_stats.accepted.fetch_add(1, Ordering::SeqCst);
                                    info!("Share accepted");
                                } else {
                                    info!("{:?} (Last: {})", msg.clone(), self.last_stratum_id.load(Ordering::SeqCst));
                                    warn!("Ignoring result for now");
                                }
                                Ok(())
                            }
                            StratumResult::Subscribe((ref _subscriptions, ref extranonce, ref nonce_size)) => {
                                self.set_extranonce(extranonce.as_str(), nonce_size)
                                /*for (name, value) in _subscriptions {
                                    match name.as_str() {
                                        "mining.set_difficulty" => {self.set_difficulty(&f32::from_str(value.as_str())?)?;},
                                        _ => {warn!("Ignored {} (={})", name, value);}
                                    }
                                }
                                Ok(())*/
                            }
                            _ => Err(format!("Inconsistent stratum message: {:?}", msg).into()),
                        }
                    }
                    StratumLinePayload::StratumCommand(command) => match command {
                        StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((
                            ref extranonce,
                            ref nonce_size,
                        ))) => self.set_extranonce(extranonce.as_str(), nonce_size),
                        StratumCommand::MiningSetDifficulty((ref difficulty,)) => self.set_difficulty(difficulty),
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyShort((id, header_hash, timestamp))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            miner
                                .process_block(Some(PartialBlock {
                                    id,
                                    header_hash,
                                    timestamp,
                                    nonce: 0,
                                    target: self.target_pool,
                                    nonce_mask: self.nonce_mask,
                                    nonce_fixed: self.nonce_fixed,
                                    hash: None,
                                }))
                                .await
                        }
                        _ => Err(format!("Unexpected stratum message: {:?}", msg).into()),
                    },
                    _ => Err(format!("Inconsistent stratum message: {:?}", msg).into()),
                }
            }
            StratumLine {
                id: Some(id),
                payload: StratumLinePayload::StratumResult { .. },
                error: Some(StratumError { code, message: error, .. }),
                ..
            } => {
                let jobid = { self.shares_stats.shares_pending.try_lock().unwrap().remove(&id) }.unwrap();
                match code {
                    ErrorCode::Unknown => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                    ErrorCode::JobNotFound => {
                        self.shares_stats.stale.fetch_add(1, Ordering::SeqCst);
                        warn!("Stale share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::DuplicateShare => {
                        self.shares_stats.duplicate.fetch_add(1, Ordering::SeqCst);
                        warn!("Duplicate share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::LowDifficultyShare => {
                        self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                        warn!("Low difficulty share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::Unauthorized => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                    ErrorCode::NotSubscribed => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                }
            }
            _ => Err(format!("Unhandled stratum response: {:?}", msg).into()),
        }
    }

    fn set_difficulty(&mut self, difficulty: &f32) -> Result<(), Error> {
        let mut buf = [0u64, 0u64, 0u64, 0u64];
        let (mantissa, exponent, _) = difficulty.recip().integer_decode();
        let new_mantissa = mantissa * DIFFICULTY_1_TARGET.0;
        let new_exponent = (DIFFICULTY_1_TARGET.1 + exponent) as u64;
        let start = (new_exponent / 64) as usize;
        let remainder = new_exponent % 64;

        buf[start] = new_mantissa << remainder; // bottom
        if start < 3 {
            buf[start + 1] = new_mantissa >> (64 - remainder); // top
        } else if new_mantissa.leading_zeros() < remainder as u32 {
            return Err("Target is too big".into());
        }

        self.target_pool = Uint256::new(buf);
        info!("Difficulty: {:?}, Target: 0x{}", difficulty, hex::encode(self.target_pool.to_be_bytes()));
        Ok(())
    }

    fn set_extranonce(&mut self, extranonce: &str, nonce_size: &u32) -> Result<(), Error> {
        self.extranonce = Some(extranonce.to_string());
        info!("Extra! {:?}", extranonce);
        self.nonce_fixed = u64::from_str_radix(extranonce, 16)? << (nonce_size * 8);
        info!("Extra Done!");
        self.nonce_mask = (1 << (nonce_size * 8)) - 1;
        Ok(())
    }

    async fn log_shares(shares_info: Arc<ShareStats>) {
        let mut ticker = tokio::time::interval(LOG_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut _last_instant = ticker.tick().await;
        loop {
            let _now = ticker.tick().await;
            info!("{}", shares_info)
        }
    }
}

impl Drop for StratumHandler {
    fn drop(&mut self) {
        self.log_handler.abort();
        self.block_handle.abort()
    }
}
