// MatrixConverter.hpp
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