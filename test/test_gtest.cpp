
#ifdef  DO_GTEST

#include <iostream>

#include "gtest/gtest.h"

TEST(sample_test_case, sample_test)
{
    EXPECT_EQ(1, 1);
}

#else

#include <iostream>


int main(){
std::cout << "hello from sequential library..." << std::endl;

return 0;

}

#endif


