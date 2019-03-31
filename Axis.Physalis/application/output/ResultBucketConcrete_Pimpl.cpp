#include "ResultBucketConcrete_Pimpl.hpp"
#include "application/output/ResultDatabase.hpp"

axis::application::output::ResultBucketConcrete::Pimpl::~Pimpl( void )
{
  while (!Databases.empty())
  {
    database_list::iterator it = Databases.begin();
    ResultDatabase *db = *it;
    db->Destroy();
    delete db;
    Databases.erase(it);
  }
}
