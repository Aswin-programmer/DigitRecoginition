#include "NumCPP.h"

template <typename T>
inline Tensor<T>::Tensor(const std::vector<size_t> &shape)
    :
    shape_(shape)
{
    if(shape_.empty())
    {
        data_.clear();
        strides_.clear();
        return;
    }

    CalculateStrides();
    data_.assign(NumElements(), T());
}

template <typename T>
inline Tensor<T>::Tensor(const std::vector<size_t> &shape
    , const std::vector<T> &data)
    :
    shape_(shape),
    data_(data)
{
    if(!shape_.empty())
    {
        CalculateStrides();
        if(data_.size() != NumElements())
        {
            std::ostringstream os;
            os << "Tensor ctor: data size("<<data_.size()<<") doesn't match shape product ("
            <<NumElements()<<")";
            throw std::runtime_error(os.str());
        }
    }
    else
    {
        if(!data_.empty() && data_.size()!=1)
        {
            throw std::runtime_error("Tensor ctor: scalar/empty-shape accepts either empty data or single element");
        }
    }
}

template <typename T>
inline std::string Tensor<T>::ToString(size_t max_elems) const
{
    std::ostringstream os;
    os << "Tensor(shape=[";
    for(size_t i=0;i<shape_.size();i++)
    {
        if(i) os<<", ";
        os<<shape_[i];
    }
    os<<"], dtype=" << typeid(T).name() << ")\n";

    size_t total = NumElements();
    size_t show = std::min(total, max_elems);
    os<<"data("<<total<<") [";
    for(size_t i=0;i<show;i++)
    {
        if(i) os<<", ";
        os<<data_[i];
    }

    if(show<total) os <<", ...";
    os <<"]\n";
    return os.str();
}

template <typename T>
inline Tensor<T> Tensor<T>::dot(const Tensor<T> &other) const
{
    // 1D · 1D -> scalar
    if (NDim() == 1 && other.NDim() == 1) {
        if (shape_[0] != other.shape_[0]) throw std::runtime_error("dot: inner dimensions must match for 1D·1D");
        T acc = T();
        for (size_t i = 0; i < shape_[0]; ++i) acc += data_[i] * other.data_[i];
        // return scalar as shape {1} for simplicity
        return Tensor<T>({1}, std::vector<T>{acc});
    }

    // 2D · 2D -> matrix multiply
    if (NDim() == 2 && other.NDim() == 2) {
        size_t M = shape_[0], K = shape_[1];
        size_t K2 = other.shape_[0], N = other.shape_[1];
        if (K != K2) throw std::runtime_error("dot: inner dimensions must match for 2D·2D");
        Tensor<T> out({M, N});
        // naive i-k-j (cache-friendly)
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                T a = data_[i * strides_[0] + k * strides_[1]]; // element A[i,k]
                for (size_t j = 0; j < N; ++j) {
                    out.data_[i * out.strides_[0] + j * out.strides_[1]] += a * other.data_[k * other.strides_[0] + j * other.strides_[1]];
                }
            }
        }
        return out;
    }

    // 2D · 1D -> result 1D of length M (matrix-vector)
    if (NDim() == 2 && other.NDim() == 1) {
        size_t M = shape_[0], K = shape_[1];
        if (K != other.shape_[0]) throw std::runtime_error("dot: inner dims must match for 2D·1D");
        Tensor<T> out({M});
        for (size_t i = 0; i < M; ++i) {
            T acc = T();
            for (size_t k = 0; k < K; ++k) {
                acc += data_[i * strides_[0] + k * strides_[1]] * other.data_[k];
            }
            out.data_[i] = acc;
        }
        return out;
    }

    // 1D · 2D -> treat vector as row (1 x K) * (K x N) -> 1 x N -> return 1D length N
    if (NDim() == 1 && other.NDim() == 2) {
        size_t K = shape_[0];
        if (K != other.shape_[0]) throw std::runtime_error("dot: inner dims must match for 1D·2D");
        size_t N = other.shape_[1];
        Tensor<T> out({N});
        for (size_t j = 0; j < N; ++j) {
            T acc = T();
            for (size_t k = 0; k < K; ++k) {
                acc += data_[k] * other.data_[k * other.strides_[0] + j * other.strides_[1]];
            }
            out.data_[j] = acc;
        }
        return out;
    }

    throw std::runtime_error("dot: unsupported operand ranks (only 1D/2D supported)");
}

template <typename T>
inline size_t Tensor<T>::NumElements() const
{
    if(shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), (size_t)1, std::multiplies<size_t>());
}

template <typename T>
inline void Tensor<T>::CalculateStrides()
{
    strides_.assign(shape_.size(), 1);
    if(shape_.empty()) return;

    for(int i=(int)shape_.size()-2;i>=0;i--)
    {
        strides_[i] = strides_[i+1] * shape_[i+1];
    }
}

template <typename T>
inline std::vector<size_t> Tensor<T>::BroadcastShapesImpl(const std::vector<size_t> &a, const std::vector<size_t> &b)
{
    size_t na = a.size();
    size_t nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<size_t> out(n, 1);
    for(size_t i=0;i<n;i++)
    {
        size_t ai = (i<n-na) ? 1 : a[i - (n-na)];
        size_t bi = (i<n-nb) ? 1 : b[i - (n-nb)];
        if(ai!=bi && ai!=1 && bi!=1)
        {
            std::ostringstream os;
            os<<"Broadcast Shapes: shapes not compatible: [";
            for(auto v : a) os<<v<<",";
            os<<"] vs [";
            for(auto v : b) os<<v<<",";
            os<<"]";
            throw std::runtime_error(os.str());
        }
        out[i] = std::max(ai, bi);
    }
    return out;
}
