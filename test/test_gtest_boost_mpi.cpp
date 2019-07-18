#include <gtest/gtest.h>
#include <boost/mpi.hpp>

class MPIEnvironment : public ::testing::Environment
{
public:
  virtual void SetUp() {
    char** argv;
    int argc = 0;
    int mpiError = MPI_Init(&argc, &argv);
    ASSERT_FALSE(mpiError);
  }
  virtual void TearDown() {
    int mpiError = MPI_Finalize();
    ASSERT_FALSE(mpiError);
  }
  virtual ~MPIEnvironment() {}
};

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}


TEST(Gtest_Boost_OMPI, AllOk)
{
  // Set up communicator
  boost::mpi::communicator boostWorld;
  if (boostWorld.rank() == 0) {
    EXPECT_EQ(100, 100);
  } else {
    EXPECT_EQ(200, 200);
  }
}


TEST(Gtest_Boost_OMPI, FailOnSlaves)
{
  // Set up communicator
  boost::mpi::communicator boostWorld;
  if (boostWorld.rank() == 0) {
    EXPECT_EQ(100, 100);
  } else {
    EXPECT_EQ(0, 200);
  }
}
