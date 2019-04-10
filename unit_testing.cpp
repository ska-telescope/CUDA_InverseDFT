 
// Copyright 2019 Seth Hall, Adam Campbell, Andrew Ensor
// High Performance Computing Research Laboratory, 
// Auckland University of Technology (AUT)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <cstdio>
#include <gtest/gtest.h>

#include "inverse_direct_fourier_transform.h"
 
// TEST(DerpTest, EqualDerp)
// {
//     ASSERT_EQ(1.0, 1.0);
// }

TEST(IDFTTest, ImageApproximatelyEqual)
{
	double threshold = 1e-5; // 0.00001
	double difference = unit_test_generate_approximate_image();
	ASSERT_LE(difference, threshold); // diff <= threshold
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
