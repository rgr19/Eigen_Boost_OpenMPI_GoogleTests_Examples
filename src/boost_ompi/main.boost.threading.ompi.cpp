#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
namespace mpi = boost::mpi;
namespace mt  = mpi::threading;

int main()
{
  mpi::environment env(mt::funneled);
  if (env.thread_level() < mt::funneled) {
     env.abort(-1);
  }
  mpi::communicator world;
  std::cout << "I am process " << world.rank() << " of " << world.size()
            << "." << std::endl;
  return 0;
}