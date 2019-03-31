#pragma once
#include "application/output/ResultBucket.hpp"

namespace axis { namespace application { namespace output {

class ResultDatabase;

/**********************************************************************************************//**
 * @class ResultBucketConcrete
 *
 * @brief Manages result recording operations for a given analysis step.
 **************************************************************************************************/
class ResultBucketConcrete : public ResultBucket
{
public:
  ResultBucketConcrete(void);
  virtual ~ResultBucketConcrete(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy(void) const;

  /**
   * Registers a new database in this archives.
   *
   * @param [in,out] database The database.
   */
  void AddDatabase(ResultDatabase& database);

  /**
   * Removes a database from this archives.
   *
   * @param [in,out] database The database.
   */
  void RemoveDatabase(ResultDatabase& database);

  /**
   * Process the result.
   *
   * @param message        The message.
   * @param numericalModel The numerical model.
   */
  virtual void PlaceResult(const axis::services::messaging::ResultMessage& message, 
                           const axis::domain::analyses::NumericalModel& numericalModel);

  /**
   * Returns information about a chain stored in this bucket.
   *
   * @param index Zero-based index of the chain.
   *
   * @return The chain metadata.
   */
  virtual ChainMetadata GetChainMetadata(int index) const;

  /**
   * Returns how many chains are stored in this bucket.
   *
   * @return The chain count.
   */
  virtual int GetChainCount(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::application::output
