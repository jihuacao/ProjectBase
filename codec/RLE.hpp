// ================================================================================================
// -*- C++ -*-
// File: rle.hpp
// Author: Guilherme R. Lampert
// Created on: 16/02/16
// Brief: Simple Run Length Encoding (RLE) in C++11.
// ================================================================================================

#ifndef PROJECT_BASE_CODEC_RLE_H
#define PROJECT_BASE_CODEC_RLE_H

// ---------
//  LICENSE
// ---------
// This software is in the public domain. Where that dedication is not recognized,
// you are granted a perpetual, irrevocable license to copy, distribute, and modify
// this file as you see fit.
//
// The source code is provided "as is", without warranty of any kind, express or implied.
// No attribution is required, but a mention about the author is appreciated.
//
// -------
//  SETUP
// -------
// #define RLE_IMPLEMENTATION in one source file before including
// this file, then use rle.hpp as a normal header file elsewhere.
//
// RLE_WORD_SIZE_16 #define controls the size of the RLE word/count.
// If not defined, use 8-bits count.

#include <cstdint>

#include <codec/Define.hpp>

namespace rle
{

// RLE encode/decode raw bytes:
int easyEncode(const std::uint8_t * input, int inSizeBytes, std::uint8_t * output, int outSizeBytes);
int easyDecode(const std::uint8_t * input, int inSizeBytes, std::uint8_t * output, int outSizeBytes);

} // namespace rle {}

// ================== End of header file ==================
#endif // RLE_HPP
// ================== End of header file ==================

