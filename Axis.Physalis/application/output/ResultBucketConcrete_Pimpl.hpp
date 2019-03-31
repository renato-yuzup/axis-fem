#pragma once
#include "ResultBucketConcrete.hpp"
#include <vector>

namespace axis { namespace application { namespace output {

class ResultDatabase;

/**********************************************************************************************//**
 * @class ResultBucketConcrete::Pimpl
 *
 * @brief Pimpl of ResultBucketConcrete.
 **************************************************************************************************/
class ResultBucketConcrete::Pimpl
{
public:
  ~Pimpl(void);

  typedef std::vector<ResultDatabase *> database_list;
  database_list Databases;
};

} } } // namespace axis::application::output
