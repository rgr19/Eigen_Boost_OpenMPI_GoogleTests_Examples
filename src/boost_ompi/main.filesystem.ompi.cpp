#include <iostream>
#include <ctime>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>


int main(int argc, char * argv[]){

  boost::filesystem::path pathname = "/usr/local/bin";
  std::cout << pathname.string() << std::endl;

  boost::mpi::environment env{argc, argv};

  boost::mpi::communicator world;

  std::cout << world.rank() << ", " << world.size() << std::endl;

return 0;

}
