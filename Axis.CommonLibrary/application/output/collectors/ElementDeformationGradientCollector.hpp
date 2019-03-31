#pragma once
#include "ElementSetCollector.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

class AXISCOMMONLIBRARY_API ElementDeformationGradientCollector : 
  public ElementSetCollector
{
public:
  /**
   * Creates a new element deformation gradient collector.
   *
   * @param targetSetName Element set name in which this collector will act.
   *
   * @return A new collector.
   */
  static ElementDeformationGradientCollector& Create(
    const axis::String& targetSetName);

  /**
   * Creates a new element deformation gradient collector.
   *
   * @param targetSetName   Element set name in which this collector will act.
   * @param customFieldName Field name to use in recordset.
   *
   * @return A new collector.
   */
  static ElementDeformationGradientCollector& Create(
    const axis::String& targetSetName, const axis::String& customFieldName);

  virtual ~ElementDeformationGradientCollector(void);
  virtual void Destroy( void ) const;
  virtual axis::String GetFieldName( void ) const;
  virtual axis::application::output::DataType GetFieldType( void ) const;
  virtual int GetMatrixFieldRowCount( void ) const;
  virtual int GetMatrixFieldColumnCount( void ) const;
  virtual void Collect( const axis::domain::elements::FiniteElement& element, 
    const axis::services::messaging::ResultMessage& message, 
    axis::application::output::recordsets::ResultRecordset& recordset );
  virtual axis::String GetFriendlyDescription( void ) const;
private:
  ElementDeformationGradientCollector(const axis::String& targetSetName,
    const axis::String& customFieldName);

  axis::String fieldName_;
};

} } } } // namespace axis::application::output::collectors
