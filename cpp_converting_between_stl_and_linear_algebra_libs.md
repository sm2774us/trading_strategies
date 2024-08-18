# __How would you convert Matrices in C++ between STL containers and the popular Linear Algebra Third Party Libraires => `Eigen`, `Blaze`, `Fastor`, `Armadillo`, `XTensor`, and `DataFrame`__

To convert matrices between __STL containers__ (like __`std::vector<std::vector<double>>`__) and popular Linear Algebra libraries like [__`Eigen`__](https://eigen.tuxfamily.org/index.php?title=Main_Page), [__`Blaze`__](https://github.com/parsa/blaze), [__`Fastor`__](https://github.com/romeric/Fastor), [__`Armadillo`__](https://arma.sourceforge.net/download.html), [__`XTensor`__](https://github.com/xtensor-stack/xtensor), and [__`DataFrame`__](https://github.com/hosseinmoein/DataFrame), we can create a reusable, header-only library. Below is an example of how you can achieve this.

## __Header File:__ [__`MatrixConverter.hpp`__](./MatrixConverter.hpp)

```cpp
#ifndef MATRIX_CONVERTER_HPP
#define MATRIX_CONVERTER_HPP

#include <vector>
#include <Eigen/Dense>
#include <blaze/Blaze.h>
#include <Fastor/Fastor.h>
#include <armadillo>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <DataFrame/DataFrame.h>

// Conversion between STL vector and Eigen
namespace MatrixConverter {

// Convert std::vector<std::vector<double>> to Eigen::MatrixXd
Eigen::MatrixXd toEigen(const std::vector<std::vector<double>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    Eigen::MatrixXd eigen_mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            eigen_mat(i, j) = mat[i][j];
    return eigen_mat;
}

// Convert Eigen::MatrixXd to std::vector<std::vector<double>>
std::vector<std::vector<double>> fromEigen(const Eigen::MatrixXd& mat) {
    std::vector<std::vector<double>> vec(mat.rows(), std::vector<double>(mat.cols()));
    for (int i = 0; i < mat.rows(); ++i)
        for (int j = 0; j < mat.cols(); ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// Convert std::vector<std::vector<double>> to Blaze::DynamicMatrix
blaze::DynamicMatrix<double> toBlaze(const std::vector<std::vector<double>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    blaze::DynamicMatrix<double> blaze_mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            blaze_mat(i, j) = mat[i][j];
    return blaze_mat;
}

// Convert Blaze::DynamicMatrix to std::vector<std::vector<double>>
std::vector<std::vector<double>> fromBlaze(const blaze::DynamicMatrix<double>& mat) {
    std::vector<std::vector<double>> vec(mat.rows(), std::vector<double>(mat.columns()));
    for (size_t i = 0; i < mat.rows(); ++i)
        for (size_t j = 0; j < mat.columns(); ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// Convert std::vector<std::vector<double>> to Fastor::Tensor
Fastor::Tensor<double, -1, -1> toFastor(const std::vector<std::vector<double>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    Fastor::Tensor<double, -1, -1> fastor_mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            fastor_mat(i, j) = mat[i][j];
    return fastor_mat;
}

// Convert Fastor::Tensor to std::vector<std::vector<double>>
std::vector<std::vector<double>> fromFastor(const Fastor::Tensor<double, -1, -1>& mat) {
    std::vector<std::vector<double>> vec(mat.dimension(0), std::vector<double>(mat.dimension(1)));
    for (size_t i = 0; i < mat.dimension(0); ++i)
        for (size_t j = 0; j < mat.dimension(1); ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// Convert std::vector<std::vector<double>> to Armadillo::mat
arma::mat toArmadillo(const std::vector<std::vector<double>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    arma::mat arma_mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            arma_mat(i, j) = mat[i][j];
    return arma_mat;
}

// Convert Armadillo::mat to std::vector<std::vector<double>>
std::vector<std::vector<double>> fromArmadillo(const arma::mat& mat) {
    std::vector<std::vector<double>> vec(mat.n_rows, std::vector<double>(mat.n_cols));
    for (size_t i = 0; i < mat.n_rows; ++i)
        for (size_t j = 0; j < mat.n_cols; ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// Convert std::vector<std::vector<double>> to xt::xarray
xt::xarray<double> toXTensor(const std::vector<std::vector<double>>& mat) {
    std::vector<size_t> shape = {mat.size(), mat[0].size()};
    return xt::adapt(mat[0].data(), mat.size() * mat[0].size(), xt::no_ownership(), shape);
}

// Convert xt::xarray to std::vector<std::vector<double>>
std::vector<std::vector<double>> fromXTensor(const xt::xarray<double>& mat) {
    std::vector<std::vector<double>> vec(mat.shape()[0], std::vector<double>(mat.shape()[1]));
    for (size_t i = 0; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < mat.shape()[1]; ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// Convert std::vector<std::vector<double>> to hmdf::StdDataFrame
hmdf::StdDataFrame<double> toDataFrame(const std::vector<std::vector<double>>& mat) {
    hmdf::StdDataFrame<double> df;
    for (size_t i = 0; i < mat[0].size(); ++i) {
        std::vector<double> column(mat.size());
        for (size_t j = 0; j < mat.size(); ++j) {
            column[j] = mat[j][i];
        }
        df.load_data(std::make_pair(i, column));
    }
    return df;
}

// Convert hmdf::StdDataFrame to std::vector<std::vector<double>>
std::vector<std::vector<double>> fromDataFrame(const hmdf::StdDataFrame<double>& df) {
    std::vector<std::vector<double>> vec(df.get_index().size(), std::vector<double>(df.get_col_count()));
    for (size_t i = 0; i < df.get_index().size(); ++i) {
        for (size_t j = 0; j < df.get_col_count(); ++j) {
            vec[i][j] = df.get_column<double>(j)[i];
        }
    }
    return vec;
}

} // namespace MatrixConverter

#endif // MATRIX_CONVERTER_HPP
```

### **Explanation:**

1. **Eigen Conversion:**
   - Converts between `std::vector<std::vector<double>>` and `Eigen::MatrixXd`.
  
2. **Blaze Conversion:**
   - Converts between `std::vector<std::vector<double>>` and `blaze::DynamicMatrix<double>`.

3. **Fastor Conversion:**
   - Converts between `std::vector<std::vector<double>>` and `Fastor::Tensor<double, -1, -1>`.

4. **Armadillo Conversion:**
   - Converts between `std::vector<std::vector<double>>` and `arma::mat`.

5. **XTensor Conversion:**
   - Converts between `std::vector<std::vector<double>>` and `xt::xarray<double>`.

6. **DataFrame Conversion:**
   - Converts between `std::vector<std::vector<double>>` and `hmdf::StdDataFrame<double>`.

## __Usage Example:__

Below is the complete usage example that demonstrates conversions between `std::vector<std::vector<double>>` (STL containers) and all the mentioned Linear Algebra libraries: [__`Eigen`__](https://eigen.tuxfamily.org/index.php?title=Main_Page), [__`Blaze`__](https://github.com/parsa/blaze), [__`Fastor`__](https://github.com/romeric/Fastor), [__`Armadillo`__](https://arma.sourceforge.net/download.html), [__`XTensor`__](https://github.com/xtensor-stack/xtensor), and [__`DataFrame`__](https://github.com/hosseinmoein/DataFrame).

```cpp
#include "MatrixConverter.hpp"
#include <iostream>

int main() {
    std::vector<std::vector<double>> stl_matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    // STL to Eigen
    Eigen::MatrixXd eigen_matrix = MatrixConverter::toEigen(stl_matrix);
    std::cout << "Eigen Matrix:\n" << eigen_matrix << std::endl;

    // Eigen to STL
    std::vector<std::vector<double>> stl_from_eigen = MatrixConverter::fromEigen(eigen_matrix);
    std::cout << "STL from Eigen Matrix:\n";
    for (const auto& row : stl_from_eigen) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // STL to Blaze
    blaze::DynamicMatrix<double> blaze_matrix = MatrixConverter::toBlaze(stl_matrix);
    std::cout << "Blaze Matrix:\n" << blaze_matrix << std::endl;

    // Blaze to STL
    std::vector<std::vector<double>> stl_from_blaze = MatrixConverter::fromBlaze(blaze_matrix);
    std::cout << "STL from Blaze Matrix:\n";
    for (const auto& row : stl_from_blaze) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // STL to Fastor
    Fastor::Tensor<double, -1, -1> fastor_matrix = MatrixConverter::toFastor(stl_matrix);
    std::cout << "Fastor Matrix:\n" << fastor_matrix << std::endl;

    // Fastor to STL
    std::vector<std::vector<double>> stl_from_fastor = MatrixConverter::fromFastor(fastor_matrix);
    std::cout << "STL from Fastor Matrix:\n";
    for (const auto& row : stl_from_fastor) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // STL to Armadillo
    arma::mat arma_matrix = MatrixConverter::toArmadillo(stl_matrix);
    std::cout << "Armadillo Matrix:\n" << arma_matrix << std::endl;

    // Armadillo to STL
    std::vector<std::vector<double>> stl_from_armadillo = MatrixConverter::fromArmadillo(arma_matrix);
    std::cout << "STL from Armadillo Matrix:\n";
    for (const auto& row : stl_from_armadillo) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // STL to XTensor
    xt::xarray<double> xtensor_matrix = MatrixConverter::toXTensor(stl_matrix);
    std::cout << "XTensor Matrix:\n" << xtensor_matrix << std::endl;

    // XTensor to STL
    std::vector<std::vector<double>> stl_from_xtensor = MatrixConverter::fromXTensor(xtensor_matrix);
    std::cout << "STL from XTensor Matrix:\n";
    for (const auto& row : stl_from_xtensor) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // STL to DataFrame
    hmdf::StdDataFrame<double> df_matrix = MatrixConverter::toDataFrame(stl_matrix);
    std::cout << "DataFrame Matrix:\n";
    df_matrix.write<std::ostream, double>(std::cout);

    // DataFrame to STL
    std::vector<std::vector<double>> stl_from_dataframe = MatrixConverter::fromDataFrame(df_matrix);
    std::cout << "STL from DataFrame Matrix:\n";
    for (const auto& row : stl_from_dataframe) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

### **Explanation:**

- **STL to Eigen:** Converts the `std::vector<std::vector<double>>` to an `Eigen::MatrixXd` and then back to STL, printing both.
  
- **STL to Blaze:** Converts the `std::vector<std::vector<double>>` to a `blaze::DynamicMatrix<double>` and then back to STL, printing both.

- **STL to Fastor:** Converts the `std::vector<std::vector<double>>` to a `Fastor::Tensor<double, -1, -1>` and then back to STL, printing both.

- **STL to Armadillo:** Converts the `std::vector<std::vector<double>>` to an `arma::mat` and then back to STL, printing both.

- **STL to XTensor:** Converts the `std::vector<std::vector<double>>` to an `xt::xarray<double>` and then back to STL, printing both.

- **STL to DataFrame:** Converts the `std::vector<std::vector<double>>` to a `hmdf::StdDataFrame<double>` and then back to STL, printing both.

### **Expected Output:**

Each of the conversions will be printed to the console, showing the matrix in both its original STL form and its converted form in the respective library. The reverse conversion will also be printed, showing that the data is consistent throughout the conversion process.

## __Test Cases:__
Below are the exhaustive test cases for the conversions between STL containers and the Linear Algebra libraries using both [__`Google Test (gtest)`__](https://github.com/google/googletest) and [__`Catch2`__](https://github.com/catchorg/Catch2) __unit testing frameworks__.

### **Google Test: `MatrixConverterTests_GoogleTest.cpp`**

```cpp
#include <gtest/gtest.h>
#include "MatrixConverter.hpp"

class MatrixConverterTest : public ::testing::Test {
protected:
    std::vector<std::vector<double>> stl_matrix;

    void SetUp() override {
        stl_matrix = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
    }
};

TEST_F(MatrixConverterTest, STLToEigenAndBack) {
    Eigen::MatrixXd eigen_matrix = MatrixConverter::toEigen(stl_matrix);
    std::vector<std::vector<double>> result = MatrixConverter::fromEigen(eigen_matrix);
    ASSERT_EQ(stl_matrix, result);
}

TEST_F(MatrixConverterTest, STLToBlazeAndBack) {
    blaze::DynamicMatrix<double> blaze_matrix = MatrixConverter::toBlaze(stl_matrix);
    std::vector<std::vector<double>> result = MatrixConverter::fromBlaze(blaze_matrix);
    ASSERT_EQ(stl_matrix, result);
}

TEST_F(MatrixConverterTest, STLToFastorAndBack) {
    Fastor::Tensor<double, -1, -1> fastor_matrix = MatrixConverter::toFastor(stl_matrix);
    std::vector<std::vector<double>> result = MatrixConverter::fromFastor(fastor_matrix);
    ASSERT_EQ(stl_matrix, result);
}

TEST_F(MatrixConverterTest, STLToArmadilloAndBack) {
    arma::mat arma_matrix = MatrixConverter::toArmadillo(stl_matrix);
    std::vector<std::vector<double>> result = MatrixConverter::fromArmadillo(arma_matrix);
    ASSERT_EQ(stl_matrix, result);
}

TEST_F(MatrixConverterTest, STLToXTensorAndBack) {
    xt::xarray<double> xtensor_matrix = MatrixConverter::toXTensor(stl_matrix);
    std::vector<std::vector<double>> result = MatrixConverter::fromXTensor(xtensor_matrix);
    ASSERT_EQ(stl_matrix, result);
}

TEST_F(MatrixConverterTest, STLToDataFrameAndBack) {
    hmdf::StdDataFrame<double> df_matrix = MatrixConverter::toDataFrame(stl_matrix);
    std::vector<std::vector<double>> result = MatrixConverter::fromDataFrame(df_matrix);
    ASSERT_EQ(stl_matrix, result);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### **Catch2 Test: `MatrixConverterTests_Catch2.cpp`**

```cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "MatrixConverter.hpp"

std::vector<std::vector<double>> getSampleMatrix() {
    return {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
}

TEST_CASE("STL to Eigen and back", "[MatrixConverter]") {
    auto stl_matrix = getSampleMatrix();
    Eigen::MatrixXd eigen_matrix = MatrixConverter::toEigen(stl_matrix);
    auto result = MatrixConverter::fromEigen(eigen_matrix);
    REQUIRE(stl_matrix == result);
}

TEST_CASE("STL to Blaze and back", "[MatrixConverter]") {
    auto stl_matrix = getSampleMatrix();
    blaze::DynamicMatrix<double> blaze_matrix = MatrixConverter::toBlaze(stl_matrix);
    auto result = MatrixConverter::fromBlaze(blaze_matrix);
    REQUIRE(stl_matrix == result);
}

TEST_CASE("STL to Fastor and back", "[MatrixConverter]") {
    auto stl_matrix = getSampleMatrix();
    Fastor::Tensor<double, -1, -1> fastor_matrix = MatrixConverter::toFastor(stl_matrix);
    auto result = MatrixConverter::fromFastor(fastor_matrix);
    REQUIRE(stl_matrix == result);
}

TEST_CASE("STL to Armadillo and back", "[MatrixConverter]") {
    auto stl_matrix = getSampleMatrix();
    arma::mat arma_matrix = MatrixConverter::toArmadillo(stl_matrix);
    auto result = MatrixConverter::fromArmadillo(arma_matrix);
    REQUIRE(stl_matrix == result);
}

TEST_CASE("STL to XTensor and back", "[MatrixConverter]") {
    auto stl_matrix = getSampleMatrix();
    xt::xarray<double> xtensor_matrix = MatrixConverter::toXTensor(stl_matrix);
    auto result = MatrixConverter::fromXTensor(xtensor_matrix);
    REQUIRE(stl_matrix == result);
}

TEST_CASE("STL to DataFrame and back", "[MatrixConverter]") {
    auto stl_matrix = getSampleMatrix();
    hmdf::StdDataFrame<double> df_matrix = MatrixConverter::toDataFrame(stl_matrix);
    auto result = MatrixConverter::fromDataFrame(df_matrix);
    REQUIRE(stl_matrix == result);
}
```

### **Explanation:**

- **Google Test Framework:**
  - The `MatrixConverterTest` class is a fixture that holds the sample matrix used across multiple test cases.
  - Each test case checks the conversion to a specific library's matrix format and then back to STL, ensuring the data remains consistent.

- **Catch2 Framework:**
  - Each `TEST_CASE` function checks the conversion between STL and the respective library's matrix format, using `REQUIRE` to assert equality between the original and converted matrices.

These test files ensure the correctness of all conversions and can be compiled and run using their respective frameworks to validate the `MatrixConverter` utility.

## __Summary__

This header-only library allows seamless conversion between STL containers and various third-party linear algebra libraries,

