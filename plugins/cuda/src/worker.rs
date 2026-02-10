use crate::{Error, NonceGenEnum};
use cust::context::CurrentContext;
use cust::device::DeviceAttribute;
use cust::function::Function;
use cust::memory::{bytemuck, DeviceCopy};
use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::*;
use karlsen_miner::xoshiro256starstar::Xoshiro256StarStar;
use karlsen_miner::Worker;
use log::{error, info};
use rand::{Fill, RngCore};
use std::ffi::CString;
use std::ops::BitXor;
use std::sync::{Arc, Weak};
use tiny_keccak::Hasher;

static BPS: f32 = 1.0;

static PTX_86: &str = include_str!("../resources/karlsen-cuda-sm86.ptx");
static PTX_75: &str = include_str!("../resources/karlsen-cuda-sm75.ptx");
static PTX_61: &str = include_str!("../resources/karlsen-cuda-sm61.ptx");

pub struct Kernel<'kernel> {
    func: Arc<Function<'kernel>>,
    block_size: u32,
    grid_size: u32,
}

impl<'kernel> Kernel<'kernel> {
    pub fn new(module: Weak<Module>, name: &'kernel str) -> Result<Kernel<'kernel>, Error> {
        let func = Arc::new(unsafe {
            module.as_ptr().as_ref().unwrap().get_function(name).inspect_err(|&e| {
                error!("Error loading function: {}", e);
            })?
        });
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

        let device = CurrentContext::get_device()?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let grid_size = sm_count * func.max_active_blocks_per_multiprocessor(block_size.into(), 0)?;

        Ok(Self { func, block_size, grid_size })
    }

    pub fn get_workload(&self) -> u32 {
        self.block_size * self.grid_size
    }

    pub fn set_workload(&mut self, workload: u32) {
        self.grid_size = workload.div_ceil(self.block_size)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // Allow dead code for unused unions
pub union hash256 {
    pub word64s: [u64; 4usize],
    pub word32s: [u32; 8usize],
    pub bytes: [u8; 32usize],
    pub str_: [::std::os::raw::c_char; 32usize],
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // Allow dead code for unused unions
pub union hash512 {
    pub word64s: [u64; 8usize],
    pub word32s: [u32; 16usize],
    pub bytes: [u8; 64usize],
    pub str_: [::std::os::raw::c_char; 64usize],
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // Allow dead code for unused unions
pub union hash1024 {
    pub hash512s: [hash512; 2usize],
    pub word64s: [u64; 16usize],
    pub word32s: [u32; 32usize],
    pub bytes: [u8; 128usize],
    pub str_: [::std::os::raw::c_char; 128usize],
}

const SIZE_U32: usize = std::mem::size_of::<u32>();

pub trait HashData {
    fn new() -> Self;
    fn as_bytes(&self) -> &[u8];

    #[inline(always)]
    fn get_as_u32(&self, index: usize) -> u32 {
        u32::from_le_bytes(self.as_bytes()[index * SIZE_U32..index * SIZE_U32 + SIZE_U32].try_into().unwrap())
    }
}

#[derive(Debug)]
pub struct Hash256([u8; 32]);

impl HashData for Hash256 {
    fn new() -> Self {
        Self([0; 32])
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, DeviceCopy)]
pub struct Hash512([u8; 64]);
unsafe impl bytemuck::Zeroable for Hash512 {}

impl Default for Hash512 {
    fn default() -> Self {
        Self([0u8; 64])
    }
}

impl HashData for Hash512 {
    fn new() -> Self {
        Self([0; 64])
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl BitXor<&Hash512> for &Hash512 {
    type Output = Hash512;

    #[inline(always)]
    fn bitxor(self, rhs: &Hash512) -> Self::Output {
        let mut hash = Hash512::new();
        for i in 0..64 {
            hash.0[i] = self.0[i] ^ rhs.0[i];
        }
        hash
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, DeviceCopy)]
pub struct Hash1024([u8; 128]);
unsafe impl bytemuck::Zeroable for Hash1024 {}

impl Default for Hash1024 {
    fn default() -> Self {
        Self([0u8; 128])
    }
}

impl HashData for Hash1024 {
    fn new() -> Self {
        Self([0; 128])
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

const LIGHT_CACHE_ROUNDS: i32 = 3;
const LIGHT_CACHE_NUM_ITEMS: u32 = 1179641;
const FULL_DATASET_NUM_ITEMS: u32 = 37748717;
#[rustfmt::skip]
const SEED: [u8; 32] = [
    0xeb, 0x01, 0x63, 0xae, 0xf2, 0xab, 0x1c, 0x5a, 
    0x66, 0x31, 0x0c, 0x1c, 0x14, 0xd6, 0x0f, 0x42, 
    0x55, 0xa9, 0xb3, 0x9b, 0x0e, 0xdf, 0x26, 0x53, 
    0x98, 0x44, 0xf1, 0x17, 0xad, 0x67, 0x21, 0x19,
];

pub struct CudaGPUWorker {
    // NOTE: The order is important! context must be closed last
    khashv2_kernel: Kernel<'static>,
    stream: Stream,
    start_event: Event,
    stop_event: Event,
    _module: Arc<Module>,

    rand_state: DeviceBuffer<u64>,
    final_nonce_buff: DeviceBuffer<u64>,

    cache2: DeviceBuffer<Hash512>,
    dataset2: DeviceBuffer<Hash1024>,
    device_id: u32,
    pub workload: usize,
    _context: Context,

    random: NonceGenEnum,
}

impl Worker for CudaGPUWorker {
    fn id(&self) -> String {
        let device = CurrentContext::get_device().unwrap();
        format!("#{} ({})", self.device_id, device.name().unwrap())
    }

    fn load_block_constants(&mut self, hash_header: &[u8; 72], target: &[u64; 4]) {
        let mut hash_header_gpu = self._module.get_global::<[u8; 72]>(&CString::new("hash_header").unwrap()).unwrap();
        hash_header_gpu.copy_from(hash_header).map_err(|e| e.to_string()).unwrap();

        let mut target_gpu = self._module.get_global::<[u64; 4]>(&CString::new("target").unwrap()).unwrap();
        target_gpu.copy_from(target).map_err(|e| e.to_string()).unwrap();
    }

    #[inline(always)]
    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) {
        let func = &self.khashv2_kernel.func;
        let stream = &self.stream;
        let random: u8 = match self.random {
            NonceGenEnum::Lean => {
                self.rand_state.copy_from(&[rand::thread_rng().next_u64()]).unwrap();
                0
            }
            NonceGenEnum::Xoshiro => 1,
        };

        self.start_event.record(stream).unwrap();
        unsafe {
            launch!(
                func<<<
                    self.khashv2_kernel.grid_size, self.khashv2_kernel.block_size,
                    0, stream
                >>>(
                    nonce_mask,
                    nonce_fixed,
                    self.workload,
                    random,
                    self.rand_state.as_device_ptr(),
                    self.final_nonce_buff.as_device_ptr(),
                    self.dataset2.as_device_ptr(),
                    self.cache2.as_device_ptr(),
                )
            )
            .unwrap(); // We see errors in sync
        }
        self.stop_event.record(stream).unwrap();
    }

    #[inline(always)]
    fn sync(&self) -> Result<(), Error> {
        //self.stream.synchronize()?;
        self.stop_event.synchronize()?;
        if self.stop_event.elapsed_time_f32(&self.start_event)? > 1000. / BPS {
            return Err("Cuda takes longer then block rate. Please reduce your workload.".into());
        }
        Ok(())
    }

    fn get_workload(&self) -> usize {
        self.workload
    }

    #[inline(always)]
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        self.final_nonce_buff.copy_to(nonces)?;
        Ok(())
    }
}

pub fn keccak_in_place(data: &mut [u8]) {
    let mut hasher = tiny_keccak::Keccak::v512();
    hasher.update(data);
    hasher.finalize(data);
}

pub fn keccak(out: &mut [u8], data: &[u8]) {
    let mut hasher = tiny_keccak::Keccak::v512();
    hasher.update(data);
    hasher.finalize(out);
}

fn build_light_cache_cpu(cache: &mut [Hash512], seed: [u8; 32]) {
    let mut item: Hash512 = Hash512::new();
    keccak(&mut item.0, &seed);
    cache[0] = item;

    for cache_item in cache.iter_mut().take(LIGHT_CACHE_NUM_ITEMS as usize).skip(1) {
        keccak_in_place(&mut item.0);
        *cache_item = item;
    }

    for _ in 0..LIGHT_CACHE_ROUNDS {
        for i in 0..LIGHT_CACHE_NUM_ITEMS {
            // First index: 4 first bytes of the item as little-endian integer
            let t: u32 = cache[i as usize].get_as_u32(0);
            let v: u32 = t % LIGHT_CACHE_NUM_ITEMS;

            // Second index
            let w: u32 = (LIGHT_CACHE_NUM_ITEMS.wrapping_add(i.wrapping_sub(1))) % LIGHT_CACHE_NUM_ITEMS;

            let x = &cache[v as usize] ^ &cache[w as usize];
            keccak(&mut cache[i as usize].0, &x.0);
        }
    }
}

/* UNUSED */
/*
fn build_light_cache_gpu(
    cache_out: &mut DeviceBuffer<Hash512>,
    module: &Module,
    stream: &Stream,
) -> Result<(), Error> {
    let func = module.get_function("build_light_cache_gpu")?;
    let block_size = 256;
    let grid_size = (LIGHT_CACHE_NUM_ITEMS as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(func<<<grid_size, block_size, 0, stream>>>(cache_out.as_device_ptr()))?;
    }
    stream.synchronize()?;
    Ok(())
}
*/

fn build_dataset_gpu(
    dataset_out: &mut DeviceBuffer<Hash1024>,
    cache_in: &DeviceBuffer<Hash512>,
    module: &Module,
    stream: &Stream,
) -> Result<(), Error> {
    let func = module.get_function("generate_full_dataset_gpu")?;
    let block_size = 256;
    let grid_size = (FULL_DATASET_NUM_ITEMS).div_ceil(block_size);
    let start = std::time::Instant::now();

    unsafe {
        launch!(
            func<<<grid_size, block_size, 0, stream>>>(
                LIGHT_CACHE_NUM_ITEMS as i64,
                cache_in.as_device_ptr(),
                FULL_DATASET_NUM_ITEMS as i64,
                dataset_out.as_device_ptr()
            )
        )?;
    }
    stream.synchronize()?;
    info!("DAG built in {:.1}s", start.elapsed().as_secs_f32());
    Ok(())
}

impl CudaGPUWorker {
    pub fn new(
        device_id: u32,
        workload: f32,
        is_absolute: bool,
        blocking_sync: bool,
        random: NonceGenEnum,
    ) -> Result<Self, Error> {
        info!("Starting a CUDA worker");
        let sync_flag = match blocking_sync {
            true => ContextFlags::SCHED_BLOCKING_SYNC,
            false => ContextFlags::SCHED_AUTO,
        };
        let device = Device::get_device(device_id).unwrap();
        let _context = Context::new(device)?;
        _context.set_flags(sync_flag)?;

        let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
        let _module: Arc<Module>;
        info!("Device #{} compute version is {}.{}", device_id, major, minor);
        if major > 8 || (major == 8 && minor >= 6) {
            _module =
                Arc::new(Module::from_ptx(PTX_86, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else if major > 7 || (major == 7 && minor >= 5) {
            _module =
                Arc::new(Module::from_ptx(PTX_75, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else if major > 6 || (major == 6 && minor >= 1) {
            _module =
                Arc::new(Module::from_ptx(PTX_61, &[ModuleJitOption::OptLevel(OptLevel::O4)]).inspect_err(|_e| {
                    error!("Error loading PTX. Make sure you have the updated driver for you devices");
                })?);
        } else {
            return Err("Cuda compute version not supported".into());
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let mut light_cache = vec![Hash512::default(); LIGHT_CACHE_NUM_ITEMS.try_into().unwrap()];
        build_light_cache_cpu(&mut light_cache, SEED);
        let cache2 = DeviceBuffer::<Hash512>::from_slice(&light_cache)?;

        // debug light cache
        info!("light_cache[10] : {:x?}", light_cache[10].0);
        info!("light_cache[42] : {:x?}", light_cache[42].0);

        info!("Generating DAG on GPU #{} VRAM, please wait ...", device_id);
        let mut dataset2 = DeviceBuffer::<Hash1024>::zeroed(FULL_DATASET_NUM_ITEMS.try_into().unwrap())?;
        build_dataset_gpu(&mut dataset2, &cache2, &_module, &stream)?;

        /*
        // debug dataset
        let mut host_dataset = vec![Hash1024::default(); FULL_DATASET_NUM_ITEMS.try_into().unwrap()];
        dataset2.copy_to(&mut host_dataset)?;
        info!("dataset[10] : {:x?}", host_dataset[10].0);
        info!("dataset[42] : {:x?}", host_dataset[42].0);
        info!("dataset[12345] : {:x?}", host_dataset[12345].0);
        */

        let mut khashv2_kernel = Kernel::new(Arc::downgrade(&_module), "khashv2_kernel")?;

        let mut chosen_workload = 0u32;
        if is_absolute {
            chosen_workload = 1;
        } else {
            let cur_workload = khashv2_kernel.get_workload();
            if chosen_workload == 0 || chosen_workload < cur_workload {
                chosen_workload = cur_workload;
            }
        }
        chosen_workload = (chosen_workload as f32 * workload) as u32;
        info!("GPU #{} Chosen workload: {}", device_id, chosen_workload);
        khashv2_kernel.set_workload(chosen_workload);

        let final_nonce_buff = vec![0u64; 1].as_slice().as_dbuf()?;

        let rand_state: DeviceBuffer<u64> = match random {
            NonceGenEnum::Xoshiro => {
                info!("Using xoshiro for nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(4 * (chosen_workload as usize)).unwrap();
                info!("GPU #{} is generating initial seed. This may take some time.", device_id);
                let mut seed = [1u64; 4];
                seed.try_fill(&mut rand::thread_rng())?;
                buffer.copy_from(
                    Xoshiro256StarStar::new(&seed)
                        .iter_jump_state()
                        .take(chosen_workload as usize)
                        .flatten()
                        .collect::<Vec<u64>>()
                        .as_slice(),
                )?;
                info!("GPU #{} initialized", device_id);
                buffer
            }
            NonceGenEnum::Lean => {
                info!("Using lean nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(1).unwrap();
                let seed = rand::thread_rng().next_u64();
                buffer.copy_from(&[seed])?;
                buffer
            }
        };
        Ok(Self {
            device_id,
            _context,
            _module,
            start_event: Event::new(EventFlags::DEFAULT)?,
            stop_event: Event::new(EventFlags::DEFAULT)?,
            workload: chosen_workload as usize,
            stream,
            rand_state,
            final_nonce_buff,
            cache2,
            dataset2,
            khashv2_kernel,
            random,
        })
    }
}
