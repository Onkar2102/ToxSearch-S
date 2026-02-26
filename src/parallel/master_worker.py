from mpi4py import MPI


def master_main(comm, size, logger):
    """Master process (rank 0). Stub -- logs and returns."""
    n_workers = size - 1
    logger.info("Master started. Workers: %d", n_workers)


def worker_main(comm, rank, size, logger):
    """Worker process (rank > 0). Stub -- logs and returns."""
    logger.info("Worker %d started. Total ranks: %d", rank, size)


def run(logger):
    """MPI entry point. Branch on rank."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        raise RuntimeError("Need at least 2 MPI ranks (1 master + 1 worker)")

    if rank == 0:
        master_main(comm, size, logger)
    else:
        worker_main(comm, rank, size, logger)

    comm.Barrier()
