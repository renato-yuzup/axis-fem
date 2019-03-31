#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "EntityCollector.hpp"
#include "AxisString.hpp"
#include "EntityCollector.hpp"
#include "application/output/DataType.hpp"

namespace axis { 

namespace domain { namespace elements {
  class FiniteElement;
} } // namespace axis::domain::elements

namespace application { namespace output { 

namespace recordsets {
  class ResultRecordset;
} // namespace axis::application::output::recordset

namespace collectors {

/**
 * Represents an entity collector specialized in capturing data
 * from an element set of the numerical model.
 *
 * @sa EntityCollector
 */
class AXISCOMMONLIBRARY_API ElementSetCollector : public EntityCollector
{
public:
  ElementSetCollector(const axis::String& targetSetName);
  virtual ~ElementSetCollector(void);

  axis::String GetTargetSetName(void) const;

  /**
   * Returns the name of the field that would be written on a recordset by this collector.
   *
   * @return The field name.
   */
  virtual axis::String GetFieldName(void) const = 0;

  /**********************************************************************************************//**
   * @brief Returns the data type that this collector captures.
   *
   * @return  The field type.
   **************************************************************************************************/
  virtual axis::application::output::DataType GetFieldType(void) const = 0;

  /**********************************************************************************************//**
   * @brief Returns how many rows the matrix this collector writes has. If collector data type
   *        is other than matrix, this value must be equal to zero.
   *
   * @return  The matrix field row count.
   **************************************************************************************************/
  virtual int GetMatrixFieldRowCount(void) const;

  /**********************************************************************************************//**
   * @brief Returns how many columns the matrix this collector writes has. If collector data type
   *        is other than matrix, this value must be equal to zero.
   *
   * @return  The matrix field column count.
   **************************************************************************************************/
  virtual int GetMatrixFieldColumnCount(void) const;

  /**********************************************************************************************//**
   * @brief Returns how many positions the vector this collector writes has. If collector data 
   *        type is other than vector, this value must be equal to zero.
   *
   * @return  The vector field length.
   **************************************************************************************************/
  virtual int GetVectorFieldLength(void) const;

  /**
   * Pushes into a recordset data collected from an element.
   *
   * @param message            The message that triggered this request.
   * @param [in,out] recordset The recordset in which data will be fed.
   * @param element            The element from which collect data.
   */
  virtual void Collect(const axis::domain::elements::FiniteElement& element,
                       const axis::services::messaging::ResultMessage& message, 
                       axis::application::output::recordsets::ResultRecordset& recordset) = 0;

  virtual bool IsOfInterest( const axis::services::messaging::ResultMessage& message ) const;
private:
  axis::String targetSetName_;
}; // ElementSetCollector

} } } } // namespace axis::application::output::collectors
