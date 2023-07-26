#pragma once
#include "kernel.cuh"
#include <iostream>
using namespace std;

#define DX 900


class CTest
{
public:
    int *a;
    int *b;
    int *c;

    void SetParameter();
    void AddNum();
    void Show();
    void Evolution();
    void ReleaseParameter();
    void TestPTX();
    void TestCuBin();
};

void CTest::SetParameter()
{
    cudaMallocManaged(&a, sizeof(int) * DX);
    cudaMallocManaged(&b, sizeof(int) * DX);
    cudaMallocManaged(&c, sizeof(int) * DX);

    for (int f = 0; f<DX; f++)
    {
        a[f] = f;
        b[f] = f + 1;
    }

}

void CTest::ReleaseParameter(){
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

void CTest::AddNum()
{

    //const char* module_file = "my_prg.ptx";
    //const char* kernel_name = "vector_add";

    //err = cuModuleLoad(&module, module_file);
    //err = cuModuleGetFunction(&function, module, kernel_name);
    AddKernel(a, b, c, DX);
}

void CTest::Show()
{
    cout << " a     b    c"  << endl;

    for (int f = 0; f<DX; f++)
    {
        //cout << a[f] << " + " << b[f] << "  = " << c[f] << endl;
    }
}

void CTest::Evolution()
{
    SetParameter();
    AddNum();
    ReleaseParameter();
    Show();
}

void CTest::TestPTX(){
    SetParameter();
    ReleaseParameter();
    Show();
}

void CTest::TestCuBin(){
    SetParameter();
    ReleaseParameter();
    Show();
}