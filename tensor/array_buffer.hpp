#ifndef PROJECT_BASE_TENSOR_ARRAY_BUFFER_H
#define PROJECT_BASE_TENSOR_ARRAY_BUFFER_H

namespace ProjectBase{
    namespace Tensor{
        class ArrayBuffer{
            public:
                explicit ArrayBuffer(void* data_ptr);
                ~ArrayBuffer();
            public:
                void* data() const;
                size_t size() const;
        };
    }
}
#endif
