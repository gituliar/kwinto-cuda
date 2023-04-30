#pragma once

#include <cuda_runtime_api.h>

#include "kwCommon.h"


namespace kw
{

enum class kStorage { ColMajor, RowMajor };


template<kDevice Device, typename Real, kStorage Storage = kStorage::RowMajor>
class Array2d
{
public:
    __host__
    Array2d() :
        m_buf{ nullptr },
        m_nCol{ 0 },
        m_nRow{ 0 }
    {};

    __device__ __host__
    const Real&
        operator()(const size_t col, const size_t row) const { return (*this)[index(col, row)]; };

    __device__ __host__
    Real&
        operator()(const size_t col, const size_t row) { return (*this)[index(col, row)]; };

    __device__ __host__
    const Real&
        operator[](size_t i) const { return *(reinterpret_cast<Real*>(m_buf) + i); };
    __device__ __host__
    Real&
        operator[](size_t i) { return *(reinterpret_cast<Real*>(m_buf) + i); };

    __host__
    Array2d<Device, Real>&
        operator=(const Array2d<kDevice::CPU, Real>& src)
    {
        if constexpr (Device == kDevice::GPU)
        {
            // CPU -> GPU
            cudaMemcpy((void *)m_buf, src.buf(), src.bufSize(), cudaMemcpyHostToDevice);
        }
        else if constexpr (Device == kDevice::GPU)
        {
            // CPU -> CPU
            //free(m_buf);
            m_buf = src.m_buf;
        }
        else
            static_assert(true, "unknown device");

        m_nCol = src.cols();
        m_nRow = src.rows();

        return *this;
    };

    __host__
    Array2d<Device, Real>&
        operator=(const Array2d<kDevice::GPU, Real>& src)
    {
        if constexpr (Device == kDevice::GPU)
        {
            // GPU -> GPU
            m_buf = src.m_buf;
        }
        else if constexpr (Device == kDevice::CPU)
        {
            // GPU -> CPU
            auto size = bufSize();
            cudaMemcpy(m_buf, src.buf(), size, cudaMemcpyDeviceToHost);
        }
        else
            static_assert(true, "unknown device");

        m_nCol = src.cols();
        m_nRow = src.rows();

        return *this;
    };


    __device__ __host__
    size_t
        index(const size_t col, const size_t row) const
    {
        if constexpr (Storage == kStorage::ColMajor)
            return col * m_nRow + row;
        else
            return col + row * m_nCol;
    };

    __host__
    Error
        init(const size_t nCol, const size_t nRow)
    {
        if (m_buf != nullptr)
            return "kw::Array2d::init: Already initialized";

        m_nCol = nCol;
        m_nRow = nRow;

        size_t size = nCol * nRow * sizeof(Real);

        if constexpr (Device == kDevice::GPU)
        {
            if (auto cuErr = cudaMalloc((void**)&m_buf, size); cuErr != cudaSuccess)
                return "kw::Array2d::init: " + fromCudaError(cuErr);
        }
        else if constexpr (Device == kDevice::CPU)
        {
            m_buf = (int8_t*)malloc(size);
        }
        else
            static_assert(true, "unknown device");

        return "";
    };

    __host__
    void
        free()
    {
        if (m_buf)
        {
            if constexpr (Device == kDevice::GPU)
                cudaFree(m_buf);
            if constexpr (Device == kDevice::CPU)
                std::free(m_buf);
            else
                static_assert(true, "unknown device");

            m_buf = nullptr;
        }

        m_nCol = m_nRow = 0;
    }

    __device__ __host__
    const void*
        buf() const { return (void *)m_buf; };
    __device__ __host__
    size_t
        bufSize() const { return size() * sizeof(Real); }


    __device__ __host__
    size_t
        cols() const { return m_nCol; }
    __device__ __host__
    size_t
        rows() const { return m_nRow; }
    __device__ __host__
    size_t
        size() const { return m_nCol * m_nRow; }

private:
    int8_t*
        m_buf;
    size_t
        m_nCol;
    size_t
        m_nRow;
};


}
