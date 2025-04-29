from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"Rank: {comm.Get_rank()}, Size: {comm.Get_size()}")
