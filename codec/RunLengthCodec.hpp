#ifndef PROJECT_BASE_CODEC_RUNLENGTHCODEC_H
#define PROJECT_BASE_CODEC_RUNLENGTHCODEC_H
#include <ProjectBase/codec/codec.hpp>
#include <boost/property_tree/ptree.hpp>

namespace ProjectBase{
    namespace Codec{
        class RunLengthCodec
            :public ProjectBase::Codec::Codec
        {
            public:
                RunLengthCodec();
                ~RunLengthCodec();
        };
    };
}
#endif