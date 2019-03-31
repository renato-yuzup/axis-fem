#pragma once
#include "domain/curves/Curve.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace curves {

class MultiLineCurve : public Curve
{
public:
	virtual ~MultiLineCurve(void);
	virtual void Destroy( void ) const;
	virtual size_t PointCount(void) const;
	virtual real GetValueAt( real xCoord ) const;
	virtual real operator []( real xCoord ) const;
	void SetPoint(size_t index, real x, real y);
  virtual bool IsGPUCapable( void ) const;
  static axis::foundation::memory::RelativePointer Create(size_t numPoints);
	virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;
  virtual CurveUpdateCommand& GetUpdateCommand( void );
  virtual int GetGPUDataSize( void ) const;
private:
  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *, void *);
	MultiLineCurve(size_t numPoints);
  virtual void DoInitGPUData( void *data, real *outputBucket );

  axis::foundation::memory::RelativePointer pointArrayPtr_;
  uint64 numPoints_;
  real *points_;
};		

} } } // namespace axis::domain::curves
