#pragma once
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu { namespace domain { namespace elements {

class ElementData
{
public:
  GPU_ONLY ElementData(void *segmentAddress, uint64 blockIndex, 
    uint64 elementBlockSize);
  GPU_ONLY ~ElementData(void);
  GPU_ONLY uint64 GetId(void) const;
  GPU_ONLY real& ArtificialEnergy(void);
  GPU_ONLY const real& ArtificialEnergy(void) const;
  GPU_ONLY real *GetOutputBuffer(void);
  GPU_ONLY const real *GetOutputBuffer(void) const;
  GPU_ONLY void *GetFormulationBlock(void);
  GPU_ONLY const void *GetFormulationBlock(void) const;
  GPU_ONLY real Density(void) const;
  GPU_ONLY real& WaveSpeed(void);
  GPU_ONLY const real& WaveSpeed(void) const;
  GPU_ONLY real& BulkModulus(void);
  GPU_ONLY const real& BulkModulus(void) const;
  GPU_ONLY real& ShearModulus(void);
  GPU_ONLY const real& ShearModulus(void) const;
  GPU_ONLY real *MaterialTensor(void);
  GPU_ONLY const real *MaterialTensor(void) const;
  GPU_ONLY void *GetMaterialBlock(void);
  GPU_ONLY const void *GetMaterialBlock(void) const;
private:
  void *elementBlockAddr_;
  void *fBlockAddr_;
  void *mBlockAddr_;
  int formulationBlockSize_;
  uint64 elementId_;
  int dofCount_;
};

} } } } // namespace axis::yuzu::domain::elements
