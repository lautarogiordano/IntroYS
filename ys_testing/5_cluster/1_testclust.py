import numba
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64, init_xoroshiro128p_states
from numba.core.errors import NumbaPerformanceWarning, NumbaDeprecationWarning
import warnings
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
import time

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)



@cuda.jit(device=True)
def double_lock(mutex, i, j):
    # first = i;
    # second = j;
    # if i > j:
    #     first = j;
    #     second = i;

    first, second = (i, j) if i < j else (j, i)
    
    while cuda.atomic.cas(mutex, first, 0, 1) != 0:
        pass
    while cuda.atomic.cas(mutex, second, 0, 1) != 0:
        pass
    
    # cuda.threadfence()


@cuda.jit(device=True)
def double_unlock(mutex, i, j):
    # cuda.threadfence()
    cuda.atomic.exch(mutex, j, 0)
    cuda.atomic.exch(mutex, i, 0)

def get_gini(w, N):
    a = cp.sort(w)
    p_cumsum = cp.cumsum(a) / cp.sum(a)
    B = cp.sum(p_cumsum) / N
    return 1 + 1 / N - 2 * B


@cuda.jit
def stream_exchange(w, r, mutex, wmin, mcs, rng_state):
    tid = cuda.threadIdx.x

    if tid >= n_agents:
        return
    
    for t in range(mcs):
        opp = int(xoroshiro128p_uniform_float32(rng_state, tid) * (n_agents))


        # Check if both agents have enough wealth to exchange
        if opp != tid and w[tid] > wmin and w[opp] > wmin:
            double_lock(mutex, tid, opp)

            dw = min(r[tid] * w[tid], r[opp] * w[opp]) 

            # Compute the probability of winning
            prob_win = xoroshiro128p_uniform_float32(rng_state, tid)
            p = 0.5 + 0.4 * (w[opp] - w[tid]) / (w[tid] + w[opp])
            # Determine the winner and loser
            dw = dw if prob_win <= p else -dw
            
            w[tid] += dw
            w[opp] -= dw

            double_unlock(mutex, tid, opp)
    
        cuda.syncthreads()

n_streams = 1000

n_agents = 1000
wmin = 1e-17
mcs = 1000

with cuda.defer_cleanup():
    # Create 10 streams
    streams = [cuda.stream() for _ in range(n_streams)]

    # Create base arrays
    wealths = [
        np.random.rand(n_agents).astype(np.float32) for i in range(n_streams)
    ]
    risks = [
        np.random.rand(n_agents).astype(np.float32) for i in range(n_streams)
    ]
    mutexes = [
        np.zeros((n_agents,), dtype=np.int32) for i in range(n_streams)
    ]

    rng_states = [
        create_xoroshiro128p_states(n_agents, seed=time.time()) for i in range(n_streams)
    ]

    for i, wi in enumerate(wealths):
        wi /= np.sum(wi)

    events_beg = []
    events_end = []
    for i, (stream, wi, ri, mutex, rng_state) in enumerate(zip(streams, wealths, risks, mutexes, rng_states)):
        with cuda.pinned(wi, ri, mutex):
            event_beg = cuda.event()
            event_end = cuda.event()
            event_beg.record(stream=stream)

            dev_w = cuda.to_device(wi, stream=stream)
            dev_r = cuda.to_device(ri, stream=stream)
            dev_m = cuda.to_device(mutex, stream=stream)

            stream_exchange[1, n_agents, stream](dev_w, dev_r, dev_m, wmin, mcs, rng_state)

            dev_w.copy_to_host(wi, stream=stream)

            event_end.record(stream=stream)
        
        events_beg.append(event_beg)
        events_end.append(event_end)
        # Ensure that the reference to the GPU arrays are deleted, this will
        # ensure garbage collection at the exit of the context.
        del dev_w, dev_r, dev_m

    time.sleep(5)
    for event_end in events_end:
        event_end.synchronize()
    
    elapsed_times = [events_beg[0].elapsed_time(event_end) for event_end in events_end]
    i_stream_last = np.argmax(elapsed_times)

    print(f"Last stream: {i_stream_last}")
    print(f"Total time {elapsed_times[i_stream_last]:.2f} ms")