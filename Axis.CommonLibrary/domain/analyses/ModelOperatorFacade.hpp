#pragma once
#include "foundation/memory/RelativePointer.hpp"
#include "Foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { namespace analyses {

class ReducedNumericalModel;
/**
 * Provides an interface to send operations on numerical model to execute in 
 * an external (non-local CPU) processing device.
**/
class AXISCOMMONLIBRARY_API ModelOperatorFacade
{
public:
  ModelOperatorFacade(void);
  virtual ~ModelOperatorFacade(void);

  /**
   * Destroys this object.
  **/
  virtual void Destroy(void) const = 0;

  /**
   * Sets numerical model on which this operator will work on.
   *
   * @param [in,out] model  The numerical model.
  **/
  void SetTargetModel(axis::foundation::memory::RelativePointer& modelPtr);

  /**
   * Calculates the global lumped mass vector.
   *
   * @param [in,out] globalMassPtr  Relative pointer, also valid in the external device, to the
   *                                global lumped mass vector.
  **/
  virtual void CalculateGlobalLumpedMass(real t, real dt) = 0;

  /**
   * Calculates the global consistent mass matrix.
   *
   * @param [in,out] globalMassPtr  Relative pointer, also valid in the external device, to the
   *                                global consistent mass matrix.
  **/
  virtual void CalculateGlobalConsistentMass(real t, real dt) = 0;

  /**
   * Calculates the global stiffness matrix.
   *
   * @param [in,out] globalStiffnessPtr Relative pointer, also valid in the external device, to the
   *                                    global stiffness matrix.
  **/
  virtual void CalculateGlobalStiffness(real t, real dt) = 0;

  /**
   * Calculates the global internal force vector.
   *
   * @param [in,out] globalIntForcePtr  Relative pointer, also valid in the external device, to the
   *                                    global internal force vector.
  **/
  virtual void CalculateGlobalInternalForce(real time, real lastTimeIncrement, 
    real nextTimeIncrement) = 0;

  /**
   * Updates strain state of the entire model.
   *
   * @param globalDisplacementPtr   Relative pointer, also valid in the external device, to the 
   *                                global displacement vector.
  **/
  virtual void UpdateStrain(real time, real lastTimeIncrement, 
    real nextTimeIncrement) = 0;

  /**
   * Updates stress state of the entire model.
  **/
  virtual void UpdateStress(real time, real lastTimeIncrement, 
    real nextTimeIncrement) = 0;

  /**
   * Updates geometry information in all elements.
   */
  virtual void UpdateGeometry(real time, real lastTimeIncrement,
    real nextTimeIncrement) = 0;

  /**
   * Updates node physical state according to the state of adjacent elements.
  **/
  virtual void UpdateNodeQuantities(void) = 0;

  virtual void InitElementBuckets(void) = 0;

  /**
   * Updates local memory with current model state in the external processing device.
  **/
  virtual void RefreshLocalMemory(void) = 0;

  virtual void Synchronize(void) = 0;
protected:
  ReducedNumericalModel& GetModel(void);
  const ReducedNumericalModel& GetModel(void) const;
  axis::foundation::memory::RelativePointer GetModelPointer(void);
  const axis::foundation::memory::RelativePointer GetModelPointer(void) const;
private:
  axis::foundation::memory::RelativePointer modelPtr_;
};

} } } // namespace axis::domain::analyses
