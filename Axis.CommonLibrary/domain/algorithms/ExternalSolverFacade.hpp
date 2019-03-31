#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/fwd/solver_fwd.hpp"
#include "foundation/computing/KernelCommand.hpp"
#include "services/messaging/CollectorEndpoint.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace analyses {
class ReducedNumericalModel;
} } } // namespace axis::domain::analyses

namespace axis { namespace domain { namespace algorithms {

void dispatch_thread(void *thisptr, void *f);

class AXISCOMMONLIBRARY_API ExternalSolverFacade : public axis::services::messaging::CollectorEndpoint
{
public:
  ExternalSolverFacade(void);
  virtual ~ExternalSolverFacade(void);

  virtual void UpdateCurves(real time) = 0;
  virtual void UpdateAccelerations(
    axis::foundation::memory::RelativePointer& globalAccelerationVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask) = 0;
  virtual void UpdateVelocities(
    axis::foundation::memory::RelativePointer& globalVelocityVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask) = 0;
  virtual void UpdateDisplacements(
    axis::foundation::memory::RelativePointer& globalDisplacementVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask) = 0;
  virtual void UpdateExternalLoads(
    axis::foundation::memory::RelativePointer& externalLoadVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask) = 0;
  virtual void UpdateLocks(
    axis::foundation::memory::RelativePointer& globalDisplacementVector,
    real time,
    axis::foundation::memory::RelativePointer& vectorMask) = 0;
  virtual void RunKernel(axis::foundation::computing::KernelCommand& kernel) = 0;
  virtual void GatherVector(axis::foundation::memory::RelativePointer& vectorPtr,
    axis::foundation::memory::RelativePointer& modelPtr) = 0;
  virtual void Synchronize(void) = 0;

  void StartResultCollectionRound(axis::domain::analyses::ReducedNumericalModel& model);
  void DispatchMessageAsync(const axis::services::messaging::Message& message);
  void EndResultCollectionRound(void);
  bool IsCollectionRoundActive(void) const;
  void FlushResultCollection(void);
private:
  class Pimpl;
  friend class Pimpl;
  friend void dispatch_thread(void *thisptr, void *f);

  /**
   * Executes preliminary steps required before a collection round can take place.
   *
   * @param [in,out] model  The reduced numerical model of the analysis.
  **/
  virtual void PrepareForCollectionRound(axis::domain::analyses::ReducedNumericalModel& model) = 0;

  Pimpl *pimpl_;
};

} } } // namespace axis::domain::algorithms
