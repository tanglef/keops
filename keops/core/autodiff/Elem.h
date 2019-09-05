#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/ElemT.h"

namespace keops {

//////////////////////////////////////////////////////////////
////     ELEMENT EXTRACTION : Elem<F,M>                   ////
//////////////////////////////////////////////////////////////

template< class F, int N, int M >
struct ElemT;

template< class F, int M >
struct Elem : UnaryOp< Elem, F, M > {
  static const int DIM = 1;
  static_assert(F::DIM > M, "Index out of bound in Elem");

  static void PrintIdString(std::stringstream &str) {
    str << "Elem";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
    *out = outF[M];
  }

  template< class V, class GRADIN >
  using DiffTF = typename F::template DiffT< V, GRADIN >;

  template< class V, class GRADIN >
  using DiffT = DiffTF< V, ElemT< GRADIN, F::DIM, M > >;
};
}