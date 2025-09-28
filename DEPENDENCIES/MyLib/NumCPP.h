#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <type_traits>

template <typename T>
class Tensor{
    public:
    Tensor() = default;
    //Create a tensor of given shape filled with default T()
    Tensor(const std::vector<size_t>& shape);
    //Create tensor of given shape with supplied data
    Tensor(const std::vector<size_t>& shape
        , const std::vector<T>& data);

    size_t Size() const {return data_.size();}
    size_t NDim() const {return shape_.size();}
    const std::vector<size_t>& Shape() const {return shape_;}
    const std::vector<size_t>& Strides() const {return strides_;}
    T* data() {return data_.empty() ? nullptr : data_.data();}
    const T* data() const {return data_.empty() ? nullptr : data_.data();}
    //Danger: no bounds check for speed; use for internal kernels
    T& operator[](size_t flat_index){return data_.at(flat_index);}
    const T& operator[](size_t flat_index) const {return data_.at(flat_index);}

    std::string ToString(size_t max_elems = 256) const;

    static Tensor<T> Zeroes(const std::vector<size_t>& shape);

    // ----- Elementwise ops (operators) -----
    Tensor<T> operator+(const Tensor<T>& other) const { return ApplyOperation(*this, other, std::function<T(T,T)>(std::plus<T>())); }
    Tensor<T> operator-(const Tensor<T>& other) const { return ApplyOperation(*this, other, std::function<T(T,T)>(std::minus<T>())); }
    Tensor<T> operator*(const Tensor<T>& other) const { return ApplyOperation(*this, other, std::function<T(T,T)>(std::multiplies<T>())); }
    Tensor<T> operator/(const Tensor<T>& other) const { return ApplyOperation(*this, other, std::function<T(T,T)>(std::divides<T>())); }

    // ----- Dot product / matrix multiply -----
    // Supports:
    // - 1D · 1D -> scalar (returned as Tensor with shape {1})
    // - 2D · 2D -> matrix multiply
    // - 2D · 1D -> result is 1D (rows)
    // - 1D · 2D -> treated as (1 x K) * (K x N) -> 1 x N -> returned as 1D tensor
    Tensor<T> dot(const Tensor<T>& other) const;

    public:

    private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    private:
    size_t  NumElements() const;
    void CalculateStrides();

    // Friend helper that implements broadcasting-aware elementwise operation
    template <typename U>
    friend Tensor<U> ApplyOperation(const Tensor<U>& a
        , const Tensor<U>& b, std::function<U(U, U)> operation);
    //Helper for the ApplyOperation
    static std::vector<size_t> BroadcastShapesImpl(const std::vector<size_t>& a
        , const std::vector<size_t>& b);
    
};

#include "NumCpp.inl"

template <typename U>
inline Tensor<U> ApplyOperation(const Tensor<U> &A, const Tensor<U> &B, std::function<U(U, U)> operation)
{
    // Handle trivial cases
    //Identical shapes->elementwise direct
    if(A.shape_ == B.shape_)
    {
        Tensor<U> out(A.shape_);
        if(A.data_.size() != out.data_.size() || B.data_.size() != out.data_.size())
        {
            throw std::runtime_error("ApplyOperation: Unexcepted size mismatch!.");
        }
        for(size_t i=0;i<out.data_.size();i++)
        {
            out.data_[i] = operation(A.data_[i], B.data_[i]);
        }
        return out;
    }

    // Compute ouput shape using right-allign broadcast(Numpy style)
    std::vector<size_t> out_shape = Tensor<U>::BroadcastShapesImpl(A.shape_, B.shape_);
    Tensor<U> out(out_shape);

    // Precompute strides and shape alligned to out-shape(right-alligned)
    size_t nd = out_shape.size();
    std::vector<size_t> a_shape_alligned(nd, 1), b_shape_alligned(nd, 1);
    std::vector<size_t> a_strides_alligned(nd, 0), b_strides_alligned(nd, 0);

    // Fill alligned shape and strides from right.]
    for(int i=0;i<(int)nd;i++)
    {
        int idx_out = (int)nd - 1 -i;
        int idx_a = (int)A.shape_.size() -1 -i;
        int idx_b = (int)B.shape_.size() -1 -i;
        if(idx_a >= 0)
        {
            a_shape_alligned[idx_out] = A.shape_[idx_a];
            a_strides_alligned[idx_out] = A.strides_[idx_a];
        }
        else
        {
            a_shape_alligned[idx_out] = 1;
            a_strides_alligned[idx_out] = 0;
        }

        if(idx_b >= 0)
        {
            b_shape_alligned[idx_out] = B.shape_[idx_b];
            b_strides_alligned[idx_out] = B.strides_[idx_b];
        }
        else
        {
            b_shape_alligned[idx_out] = 1;
            b_strides_alligned[idx_out] = 0;
        }
    }

    // Precompute multipliers to convert flat index -> multi-index quickly.
    // multipliers[d] = product_{k=d+1..nd-1} out_shape[k]
    std::vector<size_t> multipliers(nd, 1);
    for (int d = (int)nd - 2; d >= 0; --d) multipliers[d] = multipliers[d+1] * out_shape[d+1];

    size_t out_elems = out.NumElements();
    for (size_t flat = 0; flat < out_elems; ++flat) {
        // compute offsets for A and B by decoding multi-index
        size_t offset_a = 0;
        size_t offset_b = 0;
        size_t rem = flat;
        for (size_t d = 0; d < nd; ++d) {
            size_t idx = (multipliers[d] == 0) ? 0 : (rem / multipliers[d]);
            if (multipliers[d] != 0) rem = rem % multipliers[d];
            // If input dim is 1 (broadcast), index contribution is zero (stride 0)
            if (a_shape_alligned[d] != 1) offset_a += idx * a_strides_alligned[d];
            if (b_shape_alligned[d] != 1) offset_b += idx * b_strides_alligned[d];
        }
        out.data_[flat] = operation(A.data_[offset_a], B.data_[offset_b]);
    }

    return out;
}
