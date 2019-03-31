#pragma once
#include "JobRequest.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/indexed_by.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include "AxisString.hpp"

namespace axis { namespace application { namespace jobs {

class JobRequest::Pimpl
{
public:
  typedef boost::multi_index::multi_index_container< // container with multiple indexes
        axis::String,						                     // value type: String
        boost::multi_index::indexed_by<					     // which and how many indexes?
        boost::multi_index::ordered_unique<			     // 1) an ordered with unique keys indexes
        boost::multi_index::identity<axis::String>>, // how to retrieve keys: treat String as identity
        boost::multi_index::random_access<>			     // 2) a random-access index (items in list index-accessible)
        >
    > flag_set;

  axis::String masterFilename;
  axis::String baseIncludePath;
  axis::String outputFolderPath;
  flag_set flags;
};

} } } // namespace axis::application::jobs