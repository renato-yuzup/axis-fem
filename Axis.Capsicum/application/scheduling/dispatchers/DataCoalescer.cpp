#include "DataCoalescer.hpp"
#include <map>
#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/ElementSet.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "foundation/OutOfBoundsException.hpp"

namespace aasd = axis::application::scheduling::dispatchers;
namespace ada  = axis::domain::analyses;
namespace adbc = axis::domain::boundary_conditions;
namespace adc  = axis::domain::collections;
namespace adcv = axis::domain::curves;
namespace ade  = axis::domain::elements;
namespace afm  = axis::foundation::memory;
namespace afu  = axis::foundation::uuids;

aasd::DataCoalescer::DataCoalescer(void)
{
  // nothing to do here
}

aasd::DataCoalescer::~DataCoalescer(void)
{
  // nothing to do here
}

void aasd::DataCoalescer::Coalesce( ada::NumericalModel& model )
{
  ClearState();

  // indexing of these elements can be executed in parallel
  #pragma omp parallel sections
  {
    #pragma omp section // index elements
    {
      adc::ElementSet& elements = model.Elements();
      CoalesceElement(elements);
    }
    #pragma omp section // index acceleration BCs
    {
      adc::DofList& bcList = model.AppliedAccelerations();
      CoalesceBoundaryCondition(bcList, accelerationBcs_, 
        accelerationBcGroupList_);
    }
    #pragma omp section // index velocity BCs
    {
      adc::DofList& bcList = model.AppliedVelocities();
      CoalesceBoundaryCondition(bcList, velocityBcs_, velocityBcGroupList_);
    }
    #pragma omp section // index displacement BCs
    {
      adc::DofList& bcList = model.AppliedDisplacements();
      CoalesceBoundaryCondition(bcList, displacementBcs_, 
        displacementBcGroupList_);
    }
    #pragma omp section // index load BCs
    {
      adc::DofList& bcList = model.NodalLoads();
      CoalesceBoundaryCondition(bcList, loadBcs_, loadBcGroupList_);
    }
    #pragma omp section // index lock BCs
    {
      adc::DofList& bcList = model.Locks();
      CoalesceBoundaryCondition(bcList, lockBcs_, lockBcGroupList_);
    }
    #pragma omp section // index curve BCs
    {
      adc::CurveSet& curves = model.Curves();
      CoalesceCurve(curves);
    }
  }
}

int aasd::DataCoalescer::GetElementGroupCount( void ) const
{
  return (int)elementGroupList_.size();
}

size_type aasd::DataCoalescer::GetElementGroupSize( int index ) const
{
  const element_id_list::nth_index<0>::type& bucketIndex = 
    elementGroupList_.get<0>();
  const afm::RelativePointer ptr = bucketIndex[index];
  const ade::FiniteElement& fe = absref<ade::FiniteElement>(ptr);
  const element_collection::nth_index<1>::type& eidx = elements_.get<1>();
  return (int)eidx.count(boost::make_tuple(fe.GetFormulationTypeId(), 
    fe.GetMaterialTypeId()));
}

void aasd::DataCoalescer::FillElementGroupArray( void **array, int groupIndex )
{
  element_id_list::nth_index<0>::type& bucketIndex = elementGroupList_.get<0>();
  afm::RelativePointer ptr = bucketIndex[groupIndex];
  const ade::FiniteElement& fe = absref<ade::FiniteElement>(ptr);
  element_collection::nth_index<1>::type& eidx = elements_.get<1>();
  auto bucket = eidx.find(boost::make_tuple(fe.GetFormulationTypeId(), 
    fe.GetMaterialTypeId()));
  auto end = eidx.end();
  size_type idx = 0;  
  for (; bucket != end; ++bucket, ++idx)
  {
    rel_ptr ptr = *bucket;
    array[idx] = absptr<ade::FiniteElement>(ptr);
  }
}

int aasd::DataCoalescer::GetBcGroupCount( 
  adbc::BoundaryCondition::ConstraintType bcType ) const
{
  switch (bcType)
  {
  case adbc::BoundaryCondition::PrescribedAcceleration:
    return (int)accelerationBcGroupList_.size();
  case adbc::BoundaryCondition::PrescribedDisplacement:
    return (int)displacementBcGroupList_.size();
  case adbc::BoundaryCondition::PrescribedVelocity:
    return (int)velocityBcGroupList_.size();
  case adbc::BoundaryCondition::NodalLoad:
    return (int)loadBcGroupList_.size();
  case adbc::BoundaryCondition::Lock:
    return (int)lockBcGroupList_.size();
  default:
    assert(!"Unknown boundary condition type!");
    break;
  }
}

size_type aasd::DataCoalescer::GetBcGroupSize( 
  adbc::BoundaryCondition::ConstraintType bcType, int index ) const
{
  const bc_id_list *idList = nullptr;
  const bc_collection *collection = nullptr;
  switch (bcType)
  {
  case adbc::BoundaryCondition::PrescribedAcceleration:
    idList = &accelerationBcGroupList_;
    collection = &accelerationBcs_;
    break;
  case adbc::BoundaryCondition::PrescribedDisplacement:
    idList = &displacementBcGroupList_;
    collection = &displacementBcs_;
    break;
  case adbc::BoundaryCondition::PrescribedVelocity:
    idList = &velocityBcGroupList_;
    collection = &velocityBcs_;
    break;
  case adbc::BoundaryCondition::NodalLoad:
    idList = &loadBcGroupList_;
    collection = &loadBcs_;
    break;
  case adbc::BoundaryCondition::Lock:
    idList = &lockBcGroupList_;
    collection = &lockBcs_;
    break;
  default:
    assert(!"Unknown boundary condition type!");
    break;
  }
  const auto& bucketIndex = idList->get<0>();
  const bc_type * bc = bucketIndex[index];
  const auto& bcidx = collection->get<1>();
  return (int)bcidx.count(bc->GetTypeId());
}

void aasd::DataCoalescer::FillBcGroupArray( 
  adbc::BoundaryCondition::ConstraintType bcType, void **array, int groupIndex )
{
  const bc_id_list *idList = nullptr;
  const bc_collection *collection = nullptr;
  switch (bcType)
  {
  case adbc::BoundaryCondition::PrescribedAcceleration:
    idList = &accelerationBcGroupList_;
    collection = &accelerationBcs_;
    break;
  case adbc::BoundaryCondition::PrescribedDisplacement:
    idList = &displacementBcGroupList_;
    collection = &displacementBcs_;
    break;
  case adbc::BoundaryCondition::PrescribedVelocity:
    idList = &velocityBcGroupList_;
    collection = &velocityBcs_;
    break;
  case adbc::BoundaryCondition::NodalLoad:
    idList = &loadBcGroupList_;
    collection = &loadBcs_;
    break;
  case adbc::BoundaryCondition::Lock:
    idList = &lockBcGroupList_;
    collection = &lockBcs_;
    break;
  default:
    assert(!"Unknown boundary condition type!");
    break;
  }
  auto& bucketIndex = idList->get<0>();
  bc_type * bc = bucketIndex[groupIndex];
  auto& bcidx = collection->get<1>();
  auto bucket = bcidx.find(bc->GetTypeId());
  auto end = bcidx.end();
  size_type idx = 0;  
  for (; bucket != end; ++bucket, ++idx)
  {
    bc_type * ptr = *bucket;
    array[idx] = ptr;
  }
}

int aasd::DataCoalescer::GetCurveGroupCount( void ) const
{
  return (int)curveGroupList_.size();
}

size_type aasd::DataCoalescer::GetCurveGroupSize( int index ) const
{
  const auto& bucketIndex = curveGroupList_.get<0>();
  const curve_type * curve = bucketIndex[index];
  const auto& curveIdx = curves_.get<1>();
  return (int)curveIdx.count(curve->GetTypeId());
}

void aasd::DataCoalescer::FillCurveGroupArray( void **array, int groupIndex )
{
  auto& bucketIndex = curveGroupList_.get<0>();
  curve_type * curve = bucketIndex[groupIndex];
  auto& curveIdx = curves_.get<1>();
  auto bucket = curveIdx.find(curve->GetTypeId());
  auto end = curveIdx.end();
  size_type idx = 0;  
  for (; bucket != end; ++bucket, ++idx)
  {
    curve_type * ptr = *bucket;
    array[idx] = ptr;
  }
}

void aasd::DataCoalescer::ClearState( void )
{
  elements_.clear();
  elementGroupList_.clear();
  accelerationBcs_.clear();
  accelerationBcGroupList_.clear();
  velocityBcs_.clear();
  velocityBcGroupList_.clear();
  displacementBcs_.clear();
  displacementBcGroupList_.clear();
  loadBcs_.clear();
  loadBcGroupList_.clear();
  lockBcs_.clear();
  lockBcGroupList_.clear();
  curves_.clear();
  curveGroupList_.clear();
}

void aasd::DataCoalescer::CoalesceElement( adc::ElementSet& set )
{
  size_type elementCount = set.Count();
  auto& bucketIndex = elementGroupList_.get<1>();
  auto bucketIdxEnd = bucketIndex.end();
  for (size_type i = 0; i < elementCount; i++)
  {
    afm::RelativePointer ptr = set.GetPointerByPosition(i);
    ade::FiniteElement& e = absref<ade::FiniteElement>(ptr);
    afu::Uuid formulationId = e.GetFormulationTypeId();
    afu::Uuid materialId = e.GetMaterialTypeId();
    elements_.push_back(ptr);
    if (bucketIndex.find(
      boost::make_tuple(formulationId, materialId)) == bucketIdxEnd)
    {
      bucketIndex.insert(ptr);
    }
  }
}

void aasd::DataCoalescer::CoalesceBoundaryCondition( adc::DofList& bcList, 
  bc_collection& targetCollection, bc_id_list& targetBucketIndex )
{
  size_type bcCount = bcList.Count();
  auto& bucketIndex = targetBucketIndex.get<1>();
  auto bucketIdxEnd = bucketIndex.end();
  for (size_type i = 0; i < bcCount; i++)
  {
    adbc::BoundaryCondition& bc = bcList[i].GetBoundaryCondition();
    afu::Uuid bcId = bc.GetTypeId();
    targetCollection.push_back(&bc);
    if (bucketIndex.find(bcId) == bucketIdxEnd)
    {
      bucketIndex.insert(&bc);
    }
  }
}

void aasd::DataCoalescer::CoalesceCurve( adc::CurveSet& set )
{
  size_type curveCount = set.Count();
  auto& bucketIndex = curveGroupList_.get<1>();
  auto bucketIdxEnd = bucketIndex.end();
  for (size_type i = 0; i < curveCount; i++)
  {
    adcv::Curve& curve = set[i];
    afu::Uuid curveId = curve.GetTypeId();
    curves_.push_back(&curve);
    if (bucketIndex.find(curveId) == bucketIdxEnd)
    {
      bucketIndex.insert(&curve);
    }
  }
}
