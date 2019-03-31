#pragma once
#include "foundation/axis.SystemBase.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "Foundation/Axis.CommonLibrary.hpp"
namespace axis { namespace domain { namespace analyses {

class ModelOperatorFacade;

/**
 * Implements a version of the numerical model with reduced functionality,
 * so that it can be used with external processing devices.
**/
class AXISCOMMONLIBRARY_API ReducedNumericalModel
{
public:
  ReducedNumericalModel(NumericalModel& sourceModel, ModelOperatorFacade& op);
  ~ReducedNumericalModel(void);

  const ModelDynamics& Dynamics(void) const;
  ModelDynamics& Dynamics(void);
  const ModelKinematics& Kinematics(void) const;
  ModelKinematics& Kinematics(void);
  size_type GetElementCount(void) const;
  size_type GetNodeCount(void) const;
  const axis::domain::elements::FiniteElement& GetElement(size_type index) const;
  axis::domain::elements::FiniteElement& GetElement(size_type index);
  const axis::foundation::memory::RelativePointer GetElementPointer(size_type index) const;
  axis::foundation::memory::RelativePointer GetElementPointer(size_type index);
  const axis::domain::elements::Node& GetNode(size_type index) const;
  axis::domain::elements::Node& GetNode(size_type index);
  const axis::foundation::memory::RelativePointer GetNodePointer(size_type index) const;
  axis::foundation::memory::RelativePointer GetNodePointer(size_type index);

  ModelOperatorFacade& GetOperator(void);

  /**
   * Creates a facade with reduced functionality that operates on components of an existing
   * numerical model.
   *
   * @param [in,out] sourceModel  Source numerical model.
   * @param [in,out] op           The object which provides an interface to external processing 
   *                              using this model.
   *
   * @return  A pointer to the reduced numerical model, located in model memory.
  **/
  static axis::foundation::memory::RelativePointer Create(NumericalModel& sourceModel, 
                                                          ModelOperatorFacade& op);
private:
  void *operator new (size_t, void *ptr);
  void operator delete(void *, void *);

  ModelOperatorFacade *operator_;
  axis::foundation::memory::RelativePointer nodeArrayPtr_;
  axis::foundation::memory::RelativePointer elementArrayPtr_;
  axis::foundation::memory::RelativePointer outputBucketArrayPtr_;
  size_type elementCount_;
  size_type nodeCount_;
  axis::foundation::memory::RelativePointer kinematics_;
  axis::foundation::memory::RelativePointer dynamics_;
};

} } } // namespace axis::domain::analyses
