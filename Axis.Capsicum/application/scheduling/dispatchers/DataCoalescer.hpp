#pragma once
#include <vector>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/composite_key.hpp>
#include "domain/fwd/finite_element_fwd.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"
#include "domain/collections/CurveSet.hpp"
#include "domain/collections/DofList.hpp"
#include "domain/collections/ElementSet.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/curves/Curve.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "foundation/memory/pointer.hpp"

namespace axis { namespace application { namespace scheduling { 
  namespace dispatchers {

/**
 * Reads the numerical model of the active processing request and groups 
 * domain entities in distinct sets so that algorithms run consistently on
 * its corresponding data set.
 */
class DataCoalescer
{
public:
  DataCoalescer(void);
  ~DataCoalescer(void);

  /**
   * Returns how much distinct element types exist in the model (and so the 
   * number of element groups).
   *
   * @return The element group count.
   */
  int GetElementGroupCount(void) const;

  /**
   * Returns how much elements exists in an element group.
   *
   * @param index Zero-based index of the element group.
   *
   * @return The element count in the group.
   */
  size_type GetElementGroupSize(int index) const;

  /**
   * Fills an array with relative pointers to the elements in an element group.
   *
   * @param [in,out] array Array to be filled. It must be large enough to fit
   *                 relative pointers to all elements in the group.
   * @param groupIndex     Zero-based index of the element group.
   */
  void FillElementGroupArray(void **array, int groupIndex);

  /**
   * Returns how much distinct boundary condition formulation exist in the model
   * (and so the number of boundary condition groups).
   *
   * @param bcType Type of boundary condition to analyze.
   *
   * @return The boundary condition group count.
   */
  int GetBcGroupCount(
    axis::domain::boundary_conditions::BoundaryCondition::ConstraintType bcType
    ) const;

  /**
   * Returns how much boundary condition exists in a boundary condition group.
   *
   * @param bcType Type of boundary condition to analyze.
   * @param index  Zero-based index of the boundary condition group.
   *
   * @return The boundary condition count in the group.
   */
  size_type GetBcGroupSize(
    axis::domain::boundary_conditions::BoundaryCondition::ConstraintType bcType,
    int index) const;

  /**
   * Fill an array with relative pointers to the boundary conditions in a
   * boundary condition group.
   *
   * @param bcType         Type of boundary condition to analyze.
   * @param [in,out] array Array to be filled. It must be large enough to fit
   *                 relative pointers to all boundary conditions in the group.
   * @param groupIndex     Zero-based index of the boundary condition group.
   */
  void FillBcGroupArray(
    axis::domain::boundary_conditions::BoundaryCondition::ConstraintType bcType,
    void **array, int groupIndex);

  /**
   * Returns how much distinct curve types exist in the model (and so the number
   * of curve groups).
   *
   * @return The curve group count.
   */
  int GetCurveGroupCount(void) const;

  /**
   * Returns how much curves exist in a curve group.
   *
   * @param index Zero-based index of the curve group.
   *
   * @return The curve count in the group.
   */
  size_type GetCurveGroupSize(int index) const;

  /**
   * Fill an array with relative pointers to the curves in a curve group.
   *
   * @param [in,out] array Array to be filled. It must be large enough to fit
   *                 relative pointers to all curves in the group.
   * @param groupIndex     Zero-based index of the curve group.
   */
  void FillCurveGroupArray(void **array, int groupIndex);

  /**
   * Scans numerical model and classifies entities.
   *
   * @param [in,out] model The model.
   */
  void Coalesce(axis::domain::analyses::NumericalModel& model);
private:
  typedef axis::foundation::uuids::Uuid unique_identifier;
  typedef axis::domain::elements::FiniteElement fe_type;
  typedef axis::foundation::memory::RelativePointer rel_ptr;
  typedef axis::domain::boundary_conditions::BoundaryCondition bc_type;
  typedef axis::domain::curves::Curve curve_type;

  struct FormulationKeyExtractor
  {
    typedef unique_identifier result_type;

    const result_type operator ()(const rel_ptr& ptr) const
    {
      return absref<fe_type>(ptr).GetFormulationTypeId();
    }
  };

  struct MaterialKeyExtractor
  {
    typedef unique_identifier result_type;

    const result_type operator ()(const rel_ptr& ptr) const
    {
      return absref<fe_type>(ptr).GetMaterialTypeId();
    }
  };

  struct BcKeyExtractor
  {
    typedef unique_identifier result_type;

    const result_type operator ()(const bc_type *ptr) const
    {
      return ptr->GetTypeId();
    }
  };

  struct CurveKeyExtractor
  {
    typedef unique_identifier result_type;

    const result_type operator ()(const curve_type *ptr) const
    {
      return ptr->GetTypeId();
    }
  };

  typedef boost::multi_index::random_access<>  numbered_index;  // list index
  
  // element related
  typedef boost::multi_index::composite_key<rel_ptr,
    FormulationKeyExtractor,MaterialKeyExtractor> fe_key;
  typedef boost::multi_index::hashed_non_unique<fe_key> element_bucket_index;
  typedef boost::multi_index::indexed_by<numbered_index, 
    element_bucket_index> element_index;
  typedef boost::multi_index::multi_index_container<rel_ptr, 
    element_index> element_collection;
  typedef boost::multi_index::hashed_unique<fe_key> element_identifier_index;
  typedef boost::multi_index::indexed_by<numbered_index, 
    element_identifier_index> element_id_index;
  typedef boost::multi_index::multi_index_container<rel_ptr, 
    element_id_index> element_id_list;

  // boundary condition related
  typedef boost::multi_index::hashed_non_unique<BcKeyExtractor> bc_bucket_index;
  typedef boost::multi_index::indexed_by<numbered_index, bc_bucket_index> 
    bc_index;
  typedef boost::multi_index::multi_index_container<bc_type *, bc_index> 
    bc_collection;
  typedef boost::multi_index::hashed_unique<BcKeyExtractor> bc_identifier_index;
  typedef boost::multi_index::indexed_by<numbered_index, bc_identifier_index> 
    bc_uuid_index;
  typedef boost::multi_index::multi_index_container<bc_type *, bc_uuid_index> 
    bc_id_list;

  // curve related
  typedef boost::multi_index::hashed_non_unique<CurveKeyExtractor> 
    curve_bucket_index;
  typedef boost::multi_index::indexed_by<numbered_index, curve_bucket_index> 
    curve_index;
  typedef boost::multi_index::multi_index_container<curve_type *, curve_index> 
    curve_collection;
  typedef boost::multi_index::hashed_unique<CurveKeyExtractor> 
    curve_identifier_index;
  typedef boost::multi_index::indexed_by<numbered_index, curve_identifier_index> 
    curve_uuid_index;
  typedef boost::multi_index::multi_index_container<curve_type *, 
    curve_uuid_index> curve_id_list;

  void ClearState(void);
  void CoalesceElement(axis::domain::collections::ElementSet& set);
  void CoalesceBoundaryCondition(axis::domain::collections::DofList& bcList, 
    bc_collection& targetCollection, bc_id_list& targetBucketIndex);
  void CoalesceCurve(axis::domain::collections::CurveSet& set);

  element_collection elements_;
  element_id_list    elementGroupList_;

  bc_collection      accelerationBcs_;
  bc_id_list         accelerationBcGroupList_;
  bc_collection      velocityBcs_;
  bc_id_list         velocityBcGroupList_;
  bc_collection      displacementBcs_;
  bc_id_list         displacementBcGroupList_;
  bc_collection      loadBcs_;
  bc_id_list         loadBcGroupList_;
  bc_collection      lockBcs_;
  bc_id_list         lockBcGroupList_;

  curve_collection   curves_;
  curve_id_list      curveGroupList_;
};

} } } } // namespace axis::application::scheduling::dispatchers
