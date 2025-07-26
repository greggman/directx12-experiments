#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

// Include the official d3dx12.h header if available
// You can download it from Microsoft-DirectX/DirectX-Headers on GitHub
// For now, we'll use the manual implementation below

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;
using Microsoft::WRL::ComPtr;

// Forward declarations and helper enums
enum CD3DX12_DEFAULT { D3D12_DEFAULT };

// Helper structures (simplified d3dx12.h replacements)
struct CD3DX12_RESOURCE_DESC : public D3D12_RESOURCE_DESC {
    CD3DX12_RESOURCE_DESC() = default;
    explicit CD3DX12_RESOURCE_DESC(const D3D12_RESOURCE_DESC& o) : D3D12_RESOURCE_DESC(o) {}
    
    static inline CD3DX12_RESOURCE_DESC Buffer(UINT64 width, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, UINT64 alignment = 0) {
        return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_BUFFER, alignment, width, 1, 1, 1, DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags);
    }
    
private:
    inline CD3DX12_RESOURCE_DESC(
        D3D12_RESOURCE_DIMENSION dimension,
        UINT64 alignment,
        UINT64 width,
        UINT height,
        UINT16 depthOrArraySize,
        UINT16 mipLevels,
        DXGI_FORMAT format,
        UINT sampleCount,
        UINT sampleQuality,
        D3D12_TEXTURE_LAYOUT layout,
        D3D12_RESOURCE_FLAGS flags) {
        Dimension = dimension;
        Alignment = alignment;
        Width = width;
        Height = height;
        DepthOrArraySize = depthOrArraySize;
        MipLevels = mipLevels;
        Format = format;
        SampleDesc.Count = sampleCount;
        SampleDesc.Quality = sampleQuality;
        Layout = layout;
        Flags = flags;
    }
};

struct CD3DX12_HEAP_PROPERTIES : public D3D12_HEAP_PROPERTIES {
    CD3DX12_HEAP_PROPERTIES() = default;
    explicit CD3DX12_HEAP_PROPERTIES(const D3D12_HEAP_PROPERTIES& o) : D3D12_HEAP_PROPERTIES(o) {}
    
    CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE type, UINT creationNodeMask = 1, UINT nodeMask = 1) {
        Type = type;
        CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        CreationNodeMask = creationNodeMask;
        VisibleNodeMask = nodeMask;
    }
};

struct CD3DX12_RANGE : public D3D12_RANGE {
    CD3DX12_RANGE() = default;
    explicit CD3DX12_RANGE(const D3D12_RANGE& o) : D3D12_RANGE(o) {}
    CD3DX12_RANGE(SIZE_T begin, SIZE_T end) {
        Begin = begin;
        End = end;
    }
};

struct CD3DX12_RESOURCE_BARRIER : public D3D12_RESOURCE_BARRIER {
    CD3DX12_RESOURCE_BARRIER() = default;
    explicit CD3DX12_RESOURCE_BARRIER(const D3D12_RESOURCE_BARRIER& o) : D3D12_RESOURCE_BARRIER(o) {}
    
    static inline CD3DX12_RESOURCE_BARRIER Transition(
        ID3D12Resource* pResource,
        D3D12_RESOURCE_STATES stateBefore,
        D3D12_RESOURCE_STATES stateAfter,
        UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) {
        
        D3D12_RESOURCE_BARRIER result;
        ZeroMemory(&result, sizeof(result));
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        result.Flags = flags;
        result.Transition.pResource = pResource;
        result.Transition.StateBefore = stateBefore;
        result.Transition.StateAfter = stateAfter;
        result.Transition.Subresource = subresource;
        return static_cast<CD3DX12_RESOURCE_BARRIER>(result);
    }
};

struct CD3DX12_RASTERIZER_DESC : public D3D12_RASTERIZER_DESC {
    CD3DX12_RASTERIZER_DESC() = default;
    explicit CD3DX12_RASTERIZER_DESC(const D3D12_RASTERIZER_DESC& o) : D3D12_RASTERIZER_DESC(o) {}
    explicit CD3DX12_RASTERIZER_DESC(CD3DX12_DEFAULT) {
        FillMode = D3D12_FILL_MODE_SOLID;
        CullMode = D3D12_CULL_MODE_BACK;
        FrontCounterClockwise = FALSE;
        DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
        DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        DepthClipEnable = TRUE;
        MultisampleEnable = FALSE;
        AntialiasedLineEnable = FALSE;
        ForcedSampleCount = 0;
        ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    }
};

struct CD3DX12_BLEND_DESC : public D3D12_BLEND_DESC {
    CD3DX12_BLEND_DESC() = default;
    explicit CD3DX12_BLEND_DESC(const D3D12_BLEND_DESC& o) : D3D12_BLEND_DESC(o) {}
    explicit CD3DX12_BLEND_DESC(CD3DX12_DEFAULT) {
        AlphaToCoverageEnable = FALSE;
        IndependentBlendEnable = FALSE;
        const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlendDesc = {
            FALSE,FALSE,
            D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
            D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
            D3D12_LOGIC_OP_NOOP,
            D3D12_COLOR_WRITE_ENABLE_ALL,
        };
        for (UINT i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
            RenderTarget[i] = defaultRenderTargetBlendDesc;
    }
};

struct CD3DX12_SHADER_BYTECODE : public D3D12_SHADER_BYTECODE {
    CD3DX12_SHADER_BYTECODE() = default;
    explicit CD3DX12_SHADER_BYTECODE(const D3D12_SHADER_BYTECODE& o) : D3D12_SHADER_BYTECODE(o) {}
    CD3DX12_SHADER_BYTECODE(ID3DBlob* pShaderBlob) {
        pShaderBytecode = pShaderBlob->GetBufferPointer();
        BytecodeLength = pShaderBlob->GetBufferSize();
    }
};

// Helper functions
inline UINT64 GetRequiredIntermediateSize(ID3D12Resource* pDestinationResource, UINT FirstSubresource, UINT NumSubresources) {
    auto Desc = pDestinationResource->GetDesc();
    UINT64 RequiredSize = 0;
    
    ID3D12Device* pDevice = nullptr;
    HRESULT hr = pDestinationResource->GetDevice(IID_PPV_ARGS(&pDevice));
    if (FAILED(hr) || pDevice == nullptr) {
        std::cout << "Failed to get device from resource. HRESULT: 0x" << std::hex << hr << std::dec << std::endl;
        return 0;
    }
    
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, 0, nullptr, nullptr, nullptr, &RequiredSize);
    pDevice->Release();
    
    return RequiredSize;
}

inline UINT64 UpdateSubresources(
    ID3D12GraphicsCommandList* pCmdList,
    ID3D12Resource* pDestinationResource,
    ID3D12Resource* pIntermediate,
    UINT64 IntermediateOffset,
    UINT FirstSubresource,
    UINT NumSubresources,
    D3D12_SUBRESOURCE_DATA* pSrcData) {
    
    UINT64 RequiredSize = GetRequiredIntermediateSize(pDestinationResource, FirstSubresource, NumSubresources);
    
    // Minor validation
    auto Desc = pIntermediate->GetDesc();
    if (Desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || 
        Desc.Width < RequiredSize + IntermediateOffset || 
        RequiredSize > SIZE_T(-1)) {
        return 0;
    }
    
    auto DestDesc = pDestinationResource->GetDesc();
    
    ID3D12Device* pDevice = nullptr;
    pDestinationResource->GetDevice(IID_PPV_ARGS(&pDevice));
    
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> Layouts(NumSubresources);
    std::vector<UINT> NumRows(NumSubresources);
    std::vector<UINT64> RowSizesInBytes(NumSubresources);
    UINT64 TotalBytes = 0;
    
    pDevice->GetCopyableFootprints(&DestDesc, FirstSubresource, NumSubresources, IntermediateOffset, Layouts.data(), NumRows.data(), RowSizesInBytes.data(), &TotalBytes);
    pDevice->Release();
    
    BYTE* pData;
    HRESULT hr = pIntermediate->Map(0, nullptr, reinterpret_cast<void**>(&pData));
    if (FAILED(hr)) {
        return 0;
    }
    
    for (UINT i = 0; i < NumSubresources; ++i) {
        D3D12_MEMCPY_DEST DestData = { pData + Layouts[i].Offset, Layouts[i].Footprint.RowPitch, SIZE_T(Layouts[i].Footprint.RowPitch) * SIZE_T(NumRows[i]) };
        
        for (UINT z = 0; z < Layouts[i].Footprint.Depth; ++z) {
            auto pDestSlice = static_cast<BYTE*>(DestData.pData) + DestData.SlicePitch * z;
            auto pSrcSlice = static_cast<const BYTE*>(pSrcData[i].pData) + pSrcData[i].SlicePitch * LONG_PTR(z);
            for (UINT y = 0; y < NumRows[i]; ++y) {
                memcpy(pDestSlice + DestData.RowPitch * y, pSrcSlice + pSrcData[i].RowPitch * LONG_PTR(y), SIZE_T(RowSizesInBytes[i]));
            }
        }
    }
    pIntermediate->Unmap(0, nullptr);
    
    for (UINT i = 0; i < NumSubresources; ++i) {
        D3D12_TEXTURE_COPY_LOCATION Dst = {};
        Dst.pResource = pDestinationResource;
        Dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        Dst.SubresourceIndex = i + FirstSubresource;
        
        D3D12_TEXTURE_COPY_LOCATION Src = {};
        Src.pResource = pIntermediate;
        Src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        Src.PlacedFootprint = Layouts[i];
        
        pCmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, nullptr);
    }
    
    return RequiredSize;
}

// Vertex structure (no longer needed but keeping for now)
struct Vertex {
    XMFLOAT3 position;
    XMFLOAT2 texCoord;
};

class DX12TextureGather {
private:    
    // Pipeline objects
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_computePipelineState;
    
    // Synchronization objects
    HANDLE m_fenceEvent;
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue;
    
    // Texture and buffer
    ComPtr<ID3D12Resource> m_texture;
    ComPtr<ID3D12Resource> m_resultBuffer;
    ComPtr<ID3D12Resource> m_readbackBuffer;
    
    // Descriptor heaps
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;

public:
    DX12TextureGather() : m_fenceValue(1) {
    }
    
    void Initialize() {
        std::cout << "Initializing DirectX 12..." << std::endl;
        LoadPipeline();
        LoadAssets();
        std::cout << "DirectX 12 initialization complete!" << std::endl;
    }
    
    void LoadPipeline() {
        UINT dxgiFactoryFlags = 0;
        
#ifdef _DEBUG
        // Enable debug layer
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
#endif
        
        ComPtr<IDXGIFactory4> factory;
        ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));
        
        // Create device
        ComPtr<IDXGIAdapter1> hardwareAdapter;
        GetHardwareAdapter(factory.Get(), &hardwareAdapter);
        ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));
        
        // Create command queue
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));
        
        // Create descriptor heap for SRV and UAV
        {
            D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
            srvUavHeapDesc.NumDescriptors = 2; // One for texture SRV, one for buffer UAV
            srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ThrowIfFailed(m_device->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&m_srvUavHeap)));
        }
        
        ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)));
    }
    
    void LoadAssets() {
        // Create root signature for compute shader
        {
            D3D12_DESCRIPTOR_RANGE ranges[2];
            // Texture SRV
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;
            
            // Buffer UAV
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 0;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 1;
            
            D3D12_ROOT_PARAMETER rootParameters[1];
            rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            rootParameters[0].DescriptorTable.NumDescriptorRanges = 2;
            rootParameters[0].DescriptorTable.pDescriptorRanges = ranges;
            
            D3D12_STATIC_SAMPLER_DESC sampler = {};
            sampler.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
            sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.MipLODBias = 0;
            sampler.MaxAnisotropy = 0;
            sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS; // Will be updated dynamically
            sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
            sampler.MinLOD = 0.0f;
            sampler.MaxLOD = D3D12_FLOAT32_MAX;
            sampler.ShaderRegister = 0;
            sampler.RegisterSpace = 0;
            sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            
            D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
            rootSignatureDesc.NumParameters = 1;
            rootSignatureDesc.pParameters = rootParameters;
            rootSignatureDesc.NumStaticSamplers = 1;
            rootSignatureDesc.pStaticSamplers = &sampler;
            rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
            
            ComPtr<ID3DBlob> signature;
            ComPtr<ID3DBlob> error;
            ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &signature, &error));
            ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
        }
        
        // Create compute pipeline state object
        {
            ComPtr<ID3DBlob> computeShader;
            
            UINT compileFlags = 0;
#ifdef _DEBUG
            compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
            
            // Compute shader that uses gather compare
            const char* computeShaderSource = R"(
Texture2D<float> g_depthTexture : register(t0);
SamplerComparisonState g_sampler : register(s0);
RWStructuredBuffer<float4> g_resultBuffer : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    // Use GatherCmp to sample the depth texture at coordinate (0.5, 0.5)
    // with reference value 0.5 and comparison function LESS
    float4 gatheredValues = g_depthTexture.GatherCmp(g_sampler, float2(0.5, 0.5), 0.5);
    g_resultBuffer[0] = gatheredValues;
}
)";
            
            ThrowIfFailed(D3DCompile(computeShaderSource, strlen(computeShaderSource), nullptr, nullptr, nullptr, "CSMain", "cs_5_0", compileFlags, 0, &computeShader, nullptr));
            
            // Describe and create compute pipeline state object
            D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
            computePsoDesc.pRootSignature = m_rootSignature.Get();
            computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
            ThrowIfFailed(m_device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_computePipelineState)));
        }
        
        // Create command list
        ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), m_computePipelineState.Get(), IID_PPV_ARGS(&m_commandList)));
        ThrowIfFailed(m_commandList->Close());
        
        // Create 2x2 texture and result buffer
        CreateTexture();
        CreateResultBuffer();
        
        // Create synchronization objects
        {
            ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
            m_fenceValue = 1;
            
            m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            if (m_fenceEvent == nullptr) {
                ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
            }
            
            WaitForCompletion();
        }
    }
    
    void CreateTexture() {
        // Create a 2x2 depth texture with specific values
        const UINT textureWidth = 2;
        const UINT textureHeight = 2;
        const UINT texturePixelSize = 4; // 32-bit depth, 4 bytes per pixel
        
        D3D12_RESOURCE_DESC textureDesc = {};
        textureDesc.MipLevels = 1;
        textureDesc.Format = DXGI_FORMAT_R32_TYPELESS; // Use typeless format for depth texture
        textureDesc.Width = textureWidth;
        textureDesc.Height = textureHeight;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        textureDesc.DepthOrArraySize = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        
        D3D12_CLEAR_VALUE clearValue = {};
        clearValue.Format = DXGI_FORMAT_D32_FLOAT;
        clearValue.DepthStencil.Depth = 1.0f;
        clearValue.DepthStencil.Stencil = 0;
        
        ThrowIfFailed(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &textureDesc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &clearValue,
            IID_PPV_ARGS(&m_texture)));
        
        const UINT64 uploadBufferSize = GetRequiredIntermediateSize(m_texture.Get(), 0, 1);
        
        ComPtr<ID3D12Resource> textureUploadHeap;
        ThrowIfFailed(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&textureUploadHeap)));
        
        // Create 2x2 depth texture with values: 0.2, 0.4, 0.6, 0.8
        std::vector<float> depthData(textureWidth * textureHeight);
        
        // Pixel (0,0) - top-left
        depthData[0] = 0.2f;
        
        // Pixel (1,0) - top-right
        depthData[1] = 0.4f;
        
        // Pixel (0,1) - bottom-left
        depthData[2] = 0.6f;
        
        // Pixel (1,1) - bottom-right
        depthData[3] = 0.8f;
        
        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData = depthData.data();
        textureData.RowPitch = textureWidth * sizeof(float);
        textureData.SlicePitch = textureData.RowPitch * textureHeight;
        
        ThrowIfFailed(m_commandAllocator->Reset());
        ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
        
        // Transition to copy destination
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST));
        
        // Map the upload buffer and copy texture data
        float* pData;
        ThrowIfFailed(textureUploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&pData)));
        
        // Copy depth data
        const float* srcData = static_cast<const float*>(textureData.pData);
        for (UINT row = 0; row < textureHeight; ++row) {
            memcpy(pData + row * (textureData.RowPitch / sizeof(float)), 
                   srcData + row * textureWidth, 
                   textureWidth * sizeof(float));
        }
        textureUploadHeap->Unmap(0, nullptr);
        
        // Copy from upload heap to default texture
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = m_texture.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;
        
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = textureUploadHeap.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Offset = 0;
        src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
        src.PlacedFootprint.Footprint.Width = textureWidth;
        src.PlacedFootprint.Footprint.Height = textureHeight;
        src.PlacedFootprint.Footprint.Depth = 1;
        src.PlacedFootprint.Footprint.RowPitch = textureData.RowPitch;
        
        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
        
        // Create shader resource view for depth texture
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_R32_FLOAT; // Read as float in shader
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
        m_device->CreateShaderResourceView(m_texture.Get(), &srvDesc, srvHandle);
        
        ThrowIfFailed(m_commandList->Close());
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    }
    
    void CreateResultBuffer() {
        // Create buffer to store the 4 float values from gather
        const UINT bufferSize = sizeof(float) * 4;
        
        // Create the result buffer (UAV)
        ThrowIfFailed(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_resultBuffer)));
        
        // Create readback buffer
        ThrowIfFailed(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_readbackBuffer)));
        
        // Create UAV for the result buffer
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = 1;
        uavDesc.Buffer.StructureByteStride = sizeof(float) * 4;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
        UINT descriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        uavHandle.ptr += descriptorSize; // Move to second descriptor (UAV)
        
        m_device->CreateUnorderedAccessView(m_resultBuffer.Get(), nullptr, &uavDesc, uavHandle);
    }
    
    void UpdateShaderResourceViewSwizzle(UINT swizzleR, UINT swizzleG, UINT swizzleB, UINT swizzleA) {
        // Create shader resource view for depth texture with custom swizzle
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(swizzleR, swizzleG, swizzleB, swizzleA);
        srvDesc.Format = DXGI_FORMAT_R32_FLOAT; // Read as float in shader
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
        m_device->CreateShaderResourceView(m_texture.Get(), &srvDesc, srvHandle);
    }
    
    ComPtr<ID3D12RootSignature> CreateRootSignatureWithComparison(D3D12_COMPARISON_FUNC comparisonFunc) {
        D3D12_DESCRIPTOR_RANGE ranges[2];
        // Texture SRV
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 1;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].RegisterSpace = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        
        // Buffer UAV
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;
        ranges[1].BaseShaderRegister = 0;
        ranges[1].RegisterSpace = 0;
        ranges[1].OffsetInDescriptorsFromTableStart = 1;
        
        D3D12_ROOT_PARAMETER rootParameters[1];
        rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParameters[0].DescriptorTable.NumDescriptorRanges = 2;
        rootParameters[0].DescriptorTable.pDescriptorRanges = ranges;
        
        D3D12_STATIC_SAMPLER_DESC sampler = {};
        sampler.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.MipLODBias = 0;
        sampler.MaxAnisotropy = 0;
        sampler.ComparisonFunc = comparisonFunc;
        sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        sampler.MinLOD = 0.0f;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderRegister = 0;
        sampler.RegisterSpace = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        
        D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
        rootSignatureDesc.NumParameters = 1;
        rootSignatureDesc.pParameters = rootParameters;
        rootSignatureDesc.NumStaticSamplers = 1;
        rootSignatureDesc.pStaticSamplers = &sampler;
        rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        
        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;
        ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &signature, &error));
        
        ComPtr<ID3D12RootSignature> rootSignature;
        ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature)));
        
        return rootSignature;
    }
    
    void ExecuteGather(D3D12_COMPARISON_FUNC comparisonFunc, UINT swizzleR, UINT swizzleG, UINT swizzleB, UINT swizzleA) {
        // Create root signature with the specified comparison function
        ComPtr<ID3D12RootSignature> currentRootSignature = CreateRootSignatureWithComparison(comparisonFunc);
        
        // Update the SRV with the specified swizzle
        UpdateShaderResourceViewSwizzle(swizzleR, swizzleG, swizzleB, swizzleA);
        
        ThrowIfFailed(m_commandAllocator->Reset());
        ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), m_computePipelineState.Get()));
        
        m_commandList->SetComputeRootSignature(currentRootSignature.Get());
        
        ID3D12DescriptorHeap* ppHeaps[] = { m_srvUavHeap.Get() };
        m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
        m_commandList->SetComputeRootDescriptorTable(0, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());
        
        // Dispatch compute shader
        m_commandList->Dispatch(1, 1, 1);
        
        // Copy result to readback buffer
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_resultBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
        m_commandList->CopyResource(m_readbackBuffer.Get(), m_resultBuffer.Get());
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_resultBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        
        ThrowIfFailed(m_commandList->Close());
        
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
        
        WaitForCompletion();
    }
    
    void ReadResults(const std::string& swizzleDesc) {
        // Map the readback buffer to read the results
        float* pData = nullptr;
        CD3DX12_RANGE readRange(0, sizeof(float) * 4);
        ThrowIfFailed(m_readbackBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pData)));
        
        std::cout << std::fixed << std::setprecision(1);
        std::cout << pData[0] << " " << pData[1] << " " << pData[2] << " " << pData[3] << " - " << swizzleDesc << std::endl;
        
        m_readbackBuffer->Unmap(0, nullptr);
    }
    
    void WaitForCompletion() {
        const UINT64 fence = m_fenceValue;
        ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fence));
        m_fenceValue++;
        
        if (m_fence->GetCompletedValue() < fence) {
            ThrowIfFailed(m_fence->SetEventOnCompletion(fence, m_fenceEvent));
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }
    }
    
private:
    void GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter, bool requestHighPerformanceAdapter = false) {
        *ppAdapter = nullptr;
        
        ComPtr<IDXGIAdapter1> adapter;
        ComPtr<IDXGIFactory6> factory6;
        
        if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
            for (UINT adapterIndex = 0; 
                 SUCCEEDED(factory6->EnumAdapterByGpuPreference(
                     adapterIndex, 
                     requestHighPerformanceAdapter == true ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_UNSPECIFIED,
                     IID_PPV_ARGS(&adapter))); 
                 ++adapterIndex) {
                
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                    continue;
                }
                
                if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
                    break;
                }
            }
        }
        
        if (adapter.Get() == nullptr) {
            for (UINT adapterIndex = 0; SUCCEEDED(pFactory->EnumAdapters1(adapterIndex, &adapter)); ++adapterIndex) {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                    continue;
                }
                
                if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
                    break;
                }
            }
        }
        
        *ppAdapter = adapter.Detach();
    }
    
    void ThrowIfFailed(HRESULT hr) {
        if (FAILED(hr)) {
            std::cout << "DirectX operation failed with HRESULT: 0x" << std::hex << hr << std::dec << std::endl;
            throw std::runtime_error("DirectX operation failed");
        }
    }
};

// Window procedure (no longer needed but keeping for compilation)
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    return DefWindowProc(hWnd, message, wParam, lParam);
}

int main() {
    std::cout << "Starting DirectX 12 Texture Gather Swizzle Test..." << std::endl;
    
    try {
        DX12TextureGather sample;
        sample.Initialize();
        
        // Define comparison functions to test
        struct ComparisonTest {
            D3D12_COMPARISON_FUNC func;
            const char* name;
        };
        
        ComparisonTest comparisons[] = {
            { D3D12_COMPARISON_FUNC_LESS, "LESS" },
            { D3D12_COMPARISON_FUNC_GREATER, "GREATER" }
        };
        
        // Define swizzle settings to test
        struct SwizzleTest {
            UINT r, g, b, a;
            const char* description;
        };
        
        SwizzleTest swizzles[] = {
            { D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3, "default" },
            { D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1, "one one one one" },
            { D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, 
              D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, 
              D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, 
              D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, "zero zero zero zero" },
            { D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, "r r r r" },
            { D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, "g g g g" },
            { D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, "b b b b" },
            { D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3, "a a a a" },
            { D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, 
              D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, "a b g r" }
        };
        
        // Outer loop: test different comparison functions
        for (const auto& comparison : comparisons) {
            std::cout << "\nComparison mode: " << comparison.name << std::endl;
            
            // Inner loop: test different swizzle settings
            for (const auto& swizzle : swizzles) {
                sample.ExecuteGather(comparison.func, swizzle.r, swizzle.g, swizzle.b, swizzle.a);
                sample.ReadResults(swizzle.description);
            }
        }
        
        std::cout << "\nProgram completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "Press any key to continue..." << std::endl;
        system("pause");
        return -1;
    }
    catch (...) {
        std::cout << "Unknown error occurred!" << std::endl;
        std::cout << "Press any key to continue..." << std::endl;
        system("pause");
        return -1;
    }
}
