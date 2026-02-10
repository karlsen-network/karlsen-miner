use clap::Parser;
use log::LevelFilter;

use crate::Error;

#[derive(Parser, Debug)]
#[clap(name = "karlsen-miner", version, about = "A Karlsen high-performance CPU/GPU miner", term_width = 0)]
pub struct Opt {
    #[clap(short, long, help = "Enable debug logging level")]
    pub debug: bool,

    #[clap(short = 'a', long = "mining-address", help = "The Karlsen address for miner rewards")]
    pub mining_address: String,

    #[clap(
        short = 's',
        long = "karlsend-address",
        default_value = "127.0.0.1",
        help = "Specify the IP, pool, or node address of the Karlsend instance. Use stratum+tcp://, stratum+ssl:// for stratum or grpc:// for Karlsend (default: grpc://127.0.0.1)"
    )]
    pub karlsend_address: String,

    #[clap(
        long = "devfund-percent", 
        default_value = "0", 
        help = "Percentage of blocks to send to the devfund (minimum 0%)", 
        parse(try_from_str = parse_devfund_percent)
    )]
    pub devfund_percent: u16,

    #[clap(short, long, help = "karlsend port [default: Mainnet = 42110, Testnet = 42111, Devnet = 42610]")]
    port: Option<u16>,

    #[clap(long, help = "Use testnet instead of mainnet [default: false]")]
    testnet: bool,

    #[clap(long, help = "Use devnet instead of mainnet [default: false]")]
    devnet: bool,

    #[clap(
        long = "mine-when-not-synced",
        help = "Mine even when karlsend is not synced",
        long_help = "Mine even when karlsend is not synced; useful when passing `--allow-submit-block-when-not-synced` to karlsend [default: false]"
    )]
    pub mine_when_not_synced: bool,

    #[clap(skip)]
    pub devfund_address: String,
}

fn parse_devfund_percent(s: &str) -> Result<u16, &'static str> {
    let err = "devfund-percent should be formatted as XX.YY, with up to two digits after the decimal";
    let mut parts = s.split('.');

    let prefix = parts.next().ok_or(err)?;
    let postfix = parts.next().unwrap_or("0");

    if parts.next().is_some() || prefix.len() > 2 || postfix.len() > 2 {
        return Err(err);
    }

    let prefix: u16 = prefix.parse().map_err(|_| err)?;
    let postfix: u16 = postfix.parse().map_err(|_| err)?;

    if prefix >= 100 || postfix >= 100 {
        return Err(err);
    }

    Ok(prefix * 100 + postfix)
}

impl Opt {
    pub fn process(&mut self) -> Result<(), Error> {
        if self.karlsend_address.is_empty() {
            self.karlsend_address = "127.0.0.1".to_string();
        }

        if !self.karlsend_address.contains("://") {
            let port_str = self.port().to_string();
            let (karlsend, port) =
                self.karlsend_address.split_once(':').unwrap_or((self.karlsend_address.as_str(), port_str.as_str()));

            self.karlsend_address = format!("grpc://{}:{}", karlsend, port);
        }
        log::info!("karlsend address: {}", self.karlsend_address);

        let miner_network = self.mining_address.split(':').next();
        self.devfund_address = "karlsen:qzrq7v5jhsc5znvtfdg6vxg7dz5x8dqe4wrh90jkdnwehp6vr8uj7csdss2l7".to_string();
        let devfund_network = self.devfund_address.split(':').next();

        if let (Some(miner_net), Some(devfund_net)) = (miner_network, devfund_network) {
            if miner_net != devfund_net {
                self.devfund_percent = 0;
                log::info!(
                    "Mining address ({}) and devfund ({}) are not from the same network. Disabling devfund.",
                    miner_net,
                    devfund_net
                );
            }
        }
        Ok(())
    }

    fn port(&mut self) -> u16 {
        *self.port.get_or_insert({
            if self.testnet {
                42210
            } else if self.devnet {
                42610
            } else {
                42110
            }
        })
    }

    pub fn log_level(&self) -> LevelFilter {
        if self.debug {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        }
    }
}
