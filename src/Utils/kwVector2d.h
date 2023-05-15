#pragma once

#include "Utils/kwCuda.h"

#include <cstring>
#include <stdexcept>


namespace kw {

template<typename Real>
void
    transpose(const Real* src, const size_t nCol, const size_t nRow, Real* dst)
{
    // FIXME: not tested
    if (src == dst)
        return;

    for (size_t n = 0; n < nCol * nRow; ++n) {
        size_t i = n / nCol;
        size_t j = n % nCol;
        //auto& [i, j] = std::div(n, nCol);
        dst[n] = src[nCol * j + i];
    }
}

// forward declaration
template<typename Real, int Options>
class Vector2d;


enum Vector2d_Options {
    kColMajor = 0x0,
    kRowMajor = 0x1,

    kCpu      = 0x0,
    kGpu      = 0x2
};

template<typename Real, int Options>
struct traits<Vector2d<Real, Options>>
{
    static constexpr bool isColMajor = bool((Options & kRowMajor) == kColMajor);
    static constexpr bool isRowMajor = bool((Options & kRowMajor) == kRowMajor);

    static constexpr bool isCpu = bool((Options & kGpu) == kCpu);
    static constexpr bool isGpu = bool((Options & kGpu) == kGpu);

    static constexpr int order = Options & kRowMajor;
    static constexpr int device = Options & kGpu;
};


template<typename Real, int Options>
class Vector2d {
public:
    using value_type = Real;

    class CudaArg;

private:
    char*   m_buf;
    size_t  m_capacity;
    size_t  m_nCol;
    size_t  m_nRow;

public:
    __host__
    Vector2d() :
        m_buf{ nullptr },
        m_capacity{ 0 },
        m_nCol{ 0 },
        m_nRow{ 0 }
    {};

    __host__
    Vector2d(const size_t nCol, const size_t nRow) :
        m_buf{ nullptr },
        m_capacity{ 0 },
        m_nCol{ nCol },
        m_nRow{ nRow }
    {
        reserve(nCol * nRow);
    };

    // Copy Constructor
    __host__
    Vector2d(Vector2d& src)
    {
        *this = src;
    }

    // Move Constructor
    __host__
    Vector2d(Vector2d&& src)
    {
        *this = std::move(src);
    }

    // Move Assignment
    __host__
    Vector2d&
        operator=(Vector2d&& src)
    {
        if (this != &src) {
            m_buf = src.m_buf;
            m_capacity = src.m_capacity;
            m_nCol = src.m_nCol;
            m_nRow = src.m_nRow;

            src.m_buf = nullptr;
            src.m_capacity = 0;
            src.m_nCol = 0;
            src.m_nRow = 0;
        }

        return *this;
    }

    // Copy Assignment
    template<typename OtherVector2d>
    __host__
    Vector2d&
        operator=(const OtherVector2d& src)
    {
        reserve(src.cols() * src.rows());

        if constexpr (traits<Vector2d>::isGpu && traits<OtherVector2d>::isCpu) {
            // GPU <- CPU
            if constexpr (traits<Vector2d>::order == traits<OtherVector2d>::order) {
                cudaMemcpy(m_buf, src.buf(), src.sizeInBytes(), cudaMemcpyHostToDevice);
            }
            else {
                Vector2d srcTemp(src.cols(), src.rows());
                transpose(src, src.cols(), src.rows(), srcTemp);

                cudaMemcpy(m_buf, srcTemp.buf(), srcTemp.sizeInBytes(), cudaMemcpyHostToDevice);
            }
        }
        else if constexpr (traits<Vector2d>::isCpu && traits<OtherVector2d>::isGpu) {
            // CPU <- GPU
            if constexpr (traits<Vector2d>::order == traits<OtherVector2d>::order) {
                cudaMemcpy(m_buf, src.buf(), src.sizeInBytes(), cudaMemcpyDeviceToHost);
            }
            if constexpr (traits<Vector2d>::order != traits<OtherVector2d>::order) {
                Vector2d dstTemp(src.cols(), src.rows());

                cudaMemcpy(dstTemp.m_buf, src.buf(), src.sizeInBytes(), cudaMemcpyDeviceToHost);

                transpose(dstTemp, dstTemp.cols(), dstTemp.rows(), m_buf);
            }
        }
        else if constexpr (traits<Vector2d>::isCpu && traits<OtherVector2d>::isCpu) {
            // CPU <- CPU
            std::memcpy(m_buf, src.buf(), src.sizeInBytes());
        }
        else if constexpr (traits<Vector2d>::isGpu && traits<OtherVector2d>::isGpu) {
            // GPU <- GPU
            cudaMemcpy(m_buf, src.buf(), src.sizeInBytes(), cudaMemcpyDeviceToDevice);
        }

        m_nCol = src.cols();
        m_nRow = src.rows();

        return *this;
    };

    __host__
    ~Vector2d()
    {
        if (traits<Vector2d>::isCpu) {
            if (m_buf) {
                std::free(m_buf);
            }
        }
        else if (traits<Vector2d>::isGpu) {
            if (m_buf) {
                auto cudaError = cudaFree(m_buf); // cudaError != cudaSuccess
            }
        }
    };


    __device__ __host__
    Real*
        buf() { return (Real *)m_buf; };
    __device__ __host__
    const Real*
        buf() const { return (Real *)m_buf; };

    __device__ __host__
    size_t
        cols() const { return m_nCol; }

    __device__ __host__
    size_t
        rows() const { return m_nRow; }

    __device__ __host__
    size_t
        size() const { return m_nCol * m_nRow; }

    __device__ __host__
    size_t
        sizeInBytes() const { return size() * sizeof(Real); }

    // Access
    __device__ __host__
    const Real&
        operator[](const size_t idx) const { return *(reinterpret_cast<Real*>(m_buf) + idx); };
    __device__ __host__
    Real&
        operator[](const size_t idx) { return *(reinterpret_cast<Real*>(m_buf) + idx); };

    __device__ __host__
    size_t
        index(const size_t col, const size_t row) const
    {
        if constexpr (traits<Vector2d>::isColMajor)
            return col * m_nRow + row;
        else
            return col + row * m_nCol;
    };

    __device__ __host__
    const Real&
        operator()(const size_t col, const size_t row) const { return *(reinterpret_cast<Real*>(m_buf) + index(col, row)); };
    __device__ __host__
    Real&
        operator()(const size_t col, const size_t row) { return *(reinterpret_cast<Real*>(m_buf) + index(col, row)); };

    // Allocate Memory
    __host__
    void
        resize(const size_t nCol, const size_t nRow)
    {
        if (nCol * nRow > m_capacity) {
            reserve(nCol * nRow);
        }

        m_nCol = nCol;
        m_nRow = nRow;

        return;
    }

private:
    __host__
    void
        reserve(const size_t capacity)
    {
        if (traits<Vector2d>::isCpu) {
            if (m_buf) {
                std::free(m_buf);
                m_buf = nullptr;
            }

            if (capacity > 0)
                m_buf = (char*)malloc(capacity * sizeof(Real));
        }
        else if (traits<Vector2d>::isGpu) {
            if (m_buf) {
                if (auto cudaError = cudaFree(m_buf); cudaError != cudaSuccess)
                    throw std::runtime_error("Vector2d::reserve: cudaError = " + fromCudaError(cudaError));
                m_buf = nullptr;
            }

            if (capacity > 0)
                if (auto cudaError = cudaMalloc((void**)&m_buf, capacity * sizeof(Real)); cudaError != cudaSuccess)
                    throw std::runtime_error("kw::Vector2d::reserve: cudaError " + fromCudaError(cudaError));
        }

        m_capacity = capacity;
    }
};

}
