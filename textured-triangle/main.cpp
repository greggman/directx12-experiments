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

// Vertex structure
struct Vertex {
    XMFLOAT3 position;
    XMFLOAT2 texCoord;
};

class DX12TexturedTriangle {
private:
    static const UINT FrameCount = 2;
    
    // Pipeline objects
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<IDXGISwapChain3> m_swapChain;
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    ComPtr<ID3D12DescriptorHeap> m_srvHeap;
    ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    
    // Synchronization objects
    UINT m_frameIndex;
    HANDLE m_fenceEvent;
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue;
    
    // Vertex buffer
    ComPtr<ID3D12Resource> m_vertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
    
    // Texture
    ComPtr<ID3D12Resource> m_texture;
    
    // Viewport
    D3D12_VIEWPORT m_viewport;
    D3D12_RECT m_scissorRect;
    
    UINT m_rtvDescriptorSize;
    
    HWND m_hwnd;
    UINT m_width;
    UINT m_height;

public:
    DX12TexturedTriangle(UINT width, UINT height) : 
        m_frameIndex(0), m_fenceValue(1), m_width(width), m_height(height) {
        
        m_viewport.TopLeftX = 0.0f;
        m_viewport.TopLeftY = 0.0f;
        m_viewport.Width = static_cast<float>(width);
        m_viewport.Height = static_cast<float>(height);
        m_viewport.MinDepth = 0.0f;
        m_viewport.MaxDepth = 1.0f;
        
        m_scissorRect.left = 0;
        m_scissorRect.top = 0;
        m_scissorRect.right = static_cast<LONG>(width);
        m_scissorRect.bottom = static_cast<LONG>(height);
    }
    
    void Initialize(HWND hwnd) {
        std::cout << "Initializing DirectX 12..." << std::endl;
        m_hwnd = hwnd;
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
        
        // Create swap chain
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
        swapChainDesc.BufferCount = FrameCount;
        swapChainDesc.Width = m_width;
        swapChainDesc.Height = m_height;
        swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.SampleDesc.Count = 1;
        
        ComPtr<IDXGISwapChain1> swapChain;
        ThrowIfFailed(factory->CreateSwapChainForHwnd(
            m_commandQueue.Get(),
            m_hwnd,
            &swapChainDesc,
            nullptr,
            nullptr,
            &swapChain));
        
        ThrowIfFailed(factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER));
        ThrowIfFailed(swapChain.As(&m_swapChain));
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
        
        // Create descriptor heaps
        {
            D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
            rtvHeapDesc.NumDescriptors = FrameCount;
            rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
            rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));
            
            D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
            srvHeapDesc.NumDescriptors = 1;
            srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ThrowIfFailed(m_device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvHeap)));
            
            m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        }
        
        // Create frame resources
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
            
            for (UINT n = 0; n < FrameCount; n++) {
                ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
                m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
                rtvHandle.ptr += m_rtvDescriptorSize;
            }
        }
        
        ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)));
    }
    
    void LoadAssets() {
        // Create root signature
        {
            D3D12_DESCRIPTOR_RANGE ranges[1];
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            
            D3D12_ROOT_PARAMETER rootParameters[1];
            rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            rootParameters[0].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[0].DescriptorTable.pDescriptorRanges = ranges;
            
            D3D12_STATIC_SAMPLER_DESC sampler = {};
            sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.MipLODBias = 0;
            sampler.MaxAnisotropy = 0;
            sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
            sampler.MinLOD = 0.0f;
            sampler.MaxLOD = D3D12_FLOAT32_MAX;
            sampler.ShaderRegister = 0;
            sampler.RegisterSpace = 0;
            sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            
            D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
            rootSignatureDesc.NumParameters = 1;
            rootSignatureDesc.pParameters = rootParameters;
            rootSignatureDesc.NumStaticSamplers = 1;
            rootSignatureDesc.pStaticSamplers = &sampler;
            rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
            
            ComPtr<ID3DBlob> signature;
            ComPtr<ID3DBlob> error;
            ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &signature, &error));
            ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
        }
        
        // Create pipeline state object
        {
            ComPtr<ID3DBlob> vertexShader;
            ComPtr<ID3DBlob> pixelShader;
            
            UINT compileFlags = 0;
#ifdef _DEBUG
            compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
            
            // Vertex shader
            const char* vertexShaderSource = R"(
struct VSInput
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

PSInput VSMain(VSInput input)
{
    PSInput result;
    result.position = float4(input.position, 1.0);
    result.texCoord = input.texCoord;
    return result;
}
)";
            
            ThrowIfFailed(D3DCompile(vertexShaderSource, strlen(vertexShaderSource), nullptr, nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, nullptr));
            
            // Pixel shader
            const char* pixelShaderSource = R"(
Texture2D g_texture : register(t0);
SamplerState g_sampler : register(s0);

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

float4 PSMain(PSInput input) : SV_TARGET
{
    return g_texture.Sample(g_sampler, input.texCoord);
}
)";
            
            ThrowIfFailed(D3DCompile(pixelShaderSource, strlen(pixelShaderSource), nullptr, nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, nullptr));
            
            // Define input layout
            D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
            };
            
            // Describe and create graphics pipeline state object
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
            psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
            psoDesc.pRootSignature = m_rootSignature.Get();
            psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
            psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
            psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            psoDesc.DepthStencilState.DepthEnable = FALSE;
            psoDesc.DepthStencilState.StencilEnable = FALSE;
            psoDesc.SampleMask = UINT_MAX;
            psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            psoDesc.NumRenderTargets = 1;
            psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            psoDesc.SampleDesc.Count = 1;
            ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));
        }
        
        // Create command list
        ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), m_pipelineState.Get(), IID_PPV_ARGS(&m_commandList)));
        ThrowIfFailed(m_commandList->Close());
        
        // Create vertex buffer
        {
            Vertex triangleVertices[] = {
                { { 0.0f, 0.25f, 0.0f }, { 0.5f, 0.0f } },
                { { 0.25f, -0.25f, 0.0f }, { 1.0f, 1.0f } },
                { { -0.25f, -0.25f, 0.0f }, { 0.0f, 1.0f } }
            };
            
            const UINT vertexBufferSize = sizeof(triangleVertices);
            
            ThrowIfFailed(m_device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&m_vertexBuffer)));
            
            UINT8* pVertexDataBegin;
            CD3DX12_RANGE readRange(0, 0);
            ThrowIfFailed(m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
            memcpy(pVertexDataBegin, triangleVertices, sizeof(triangleVertices));
            m_vertexBuffer->Unmap(0, nullptr);
            
            m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
            m_vertexBufferView.StrideInBytes = sizeof(Vertex);
            m_vertexBufferView.SizeInBytes = vertexBufferSize;
        }
        
        // Create texture
        CreateTexture();
        
        // Create synchronization objects
        {
            ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
            m_fenceValue = 1;
            
            m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            if (m_fenceEvent == nullptr) {
                ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
            }
            
            WaitForPreviousFrame();
        }
    }
    
    void CreateTexture() {
        // Create a simple checkerboard texture
        const UINT textureWidth = 256;
        const UINT textureHeight = 256;
        const UINT texturePixelSize = 4; // RGBA
        
        D3D12_RESOURCE_DESC textureDesc = {};
        textureDesc.MipLevels = 1;
        textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDesc.Width = textureWidth;
        textureDesc.Height = textureHeight;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        textureDesc.DepthOrArraySize = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        
        ThrowIfFailed(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &textureDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
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
        
        // Generate checkerboard pattern
        std::vector<UINT8> texture(textureWidth * textureHeight * texturePixelSize);
        for (UINT y = 0; y < textureHeight; y++) {
            for (UINT x = 0; x < textureWidth; x++) {
                UINT index = (y * textureWidth + x) * texturePixelSize;
                bool checker = ((x / 32) + (y / 32)) % 2;
                UINT8 color = checker ? 255 : 64;
                texture[index] = color;     // R
                texture[index + 1] = color; // G
                texture[index + 2] = color; // B
                texture[index + 3] = 255;   // A
            }
        }
        
        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData = texture.data();
        textureData.RowPitch = textureWidth * texturePixelSize;
        textureData.SlicePitch = textureData.RowPitch * textureHeight;
        
        ThrowIfFailed(m_commandAllocator->Reset());
        ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
        
        // Map the upload buffer and copy texture data
        UINT8* pData;
        ThrowIfFailed(textureUploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&pData)));
        
        // Copy texture data row by row
        const UINT8* srcData = static_cast<const UINT8*>(textureData.pData);
        for (UINT row = 0; row < textureHeight; ++row) {
            memcpy(pData + row * textureData.RowPitch, 
                   srcData + row * textureData.RowPitch, 
                   textureWidth * texturePixelSize);
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
        src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        src.PlacedFootprint.Footprint.Width = textureWidth;
        src.PlacedFootprint.Footprint.Height = textureHeight;
        src.PlacedFootprint.Footprint.Depth = 1;
        src.PlacedFootprint.Footprint.RowPitch = textureData.RowPitch;
        
        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        
        // Create shader resource view
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = textureDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        m_device->CreateShaderResourceView(m_texture.Get(), &srvDesc, m_srvHeap->GetCPUDescriptorHandleForHeapStart());
        
        ThrowIfFailed(m_commandList->Close());
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
        
        // Don't wait here - we'll wait after synchronization objects are created
    }
    
    void Render() {
        ThrowIfFailed(m_commandAllocator->Reset());
        ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), m_pipelineState.Get()));
        
        m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
        
        ID3D12DescriptorHeap* ppHeaps[] = { m_srvHeap.Get() };
        m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
        m_commandList->SetGraphicsRootDescriptorTable(0, m_srvHeap->GetGPUDescriptorHandleForHeapStart());
        
        m_commandList->RSSetViewports(1, &m_viewport);
        m_commandList->RSSetScissorRects(1, &m_scissorRect);
        
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
        
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rtvHandle.ptr += m_frameIndex * m_rtvDescriptorSize;
        m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        
        const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
        m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
        m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
        m_commandList->DrawInstanced(3, 1, 0, 0);
        
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));
        
        ThrowIfFailed(m_commandList->Close());
        
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
        
        ThrowIfFailed(m_swapChain->Present(1, 0));
        
        WaitForPreviousFrame();
    }
    
    void WaitForPreviousFrame() {
        const UINT64 fence = m_fenceValue;
        ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fence));
        m_fenceValue++;
        
        if (m_fence->GetCompletedValue() < fence) {
            ThrowIfFailed(m_fence->SetEventOnCompletion(fence, m_fenceEvent));
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }
        
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
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

// Window procedure
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    DX12TexturedTriangle* pSample = reinterpret_cast<DX12TexturedTriangle*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
    
    switch (message) {
    case WM_CREATE:
        {
            LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
            SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
        }
        return 0;
        
    case WM_PAINT:
        if (pSample) {
            pSample->Render();
        }
        return 0;
        
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    
    return DefWindowProc(hWnd, message, wParam, lParam);
}

int main() {
    // Get the current process instance
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    int nCmdShow = SW_SHOW;
    
    std::cout << "Starting DirectX 12 Textured Triangle..." << std::endl;
    
    try {
        const UINT width = 800;
        const UINT height = 600;
        
        // Register window class
        WNDCLASSEX windowClass = { 0 };
        windowClass.cbSize = sizeof(WNDCLASSEX);
        windowClass.style = CS_HREDRAW | CS_VREDRAW;
        windowClass.lpfnWndProc = WindowProc;
        windowClass.hInstance = hInstance;
        windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
        windowClass.lpszClassName = "DXSampleClass";
        RegisterClassEx(&windowClass);
        
        RECT windowRect = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };
        AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);
        
        DX12TexturedTriangle sample(width, height);
        
        // Create window
        HWND hwnd = CreateWindow(
            windowClass.lpszClassName,
            "DirectX 12 Textured Triangle",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            windowRect.right - windowRect.left,
            windowRect.bottom - windowRect.top,
            nullptr,
            nullptr,
            hInstance,
            &sample);
        
        if (hwnd == NULL) {
            std::cout << "Failed to create window. Error: " << GetLastError() << std::endl;
            return -1;
        }
        
        sample.Initialize(hwnd);
        
        ShowWindow(hwnd, nCmdShow);
        
        // Main loop
        MSG msg = {};
        while (msg.message != WM_QUIT) {
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                sample.Render();
            }
        }
        
        std::cout << "Program completed successfully!" << std::endl;
        return static_cast<char>(msg.wParam);
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
