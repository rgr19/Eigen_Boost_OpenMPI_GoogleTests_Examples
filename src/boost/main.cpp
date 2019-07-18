#include <iostream>
#include <ctime>
#include <boost/filesystem.hpp>


int main(int argc, char * argv[]){

  boost::filesystem::path pathname = "/usr/local/bin";
  std::cout << pathname.string() << std::endl;


return 0;

}
