#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "CurveUpdateCommand.hpp"

namespace axis { namespace domain { namespace curves {

class AXISCOMMONLIBRARY_API Curve
{
public:
  typedef real (*GPUCurveOperator_t)(void *, real);

	virtual ~Curve(void);

	virtual void Destroy(void) const = 0;

	virtual real GetValueAt(real xCoord) const = 0;
	virtual real operator [](real xCoord) const = 0;

  virtual bool IsGPUCapable(void) const;

  virtual axis::foundation::uuids::Uuid GetTypeId(void) const = 0;
  virtual CurveUpdateCommand& GetUpdateCommand(void);
  virtual int GetGPUDataSize(void) const;
  void InitGPUData(void *data, real *outputBucket, const real *readOnlyMirroredOutputBucketAddr);
  const real * GetGPUValueSlotPointer(void) const;
private:
  virtual void DoInitGPUData(void *data, real *outputBucket);
  const real *outputBucket_;
};

} } } // namespace axis::domain::curves
