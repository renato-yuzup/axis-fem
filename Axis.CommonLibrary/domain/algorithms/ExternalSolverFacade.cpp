#include "ExternalSolverFacade.hpp"
#include <mutex>
#include <list>
#include "foundation/InvalidOperationException.hpp"
#include <thread>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

namespace adal = axis::domain::algorithms;
namespace ada = axis::domain::analyses;
namespace asmm = axis::services::messaging;

class adal::ExternalSolverFacade::Pimpl
{
public:
  typedef std::list<axis::services::messaging::Message *> message_queue;

  Pimpl(void) : collectionRoundSemaphore(1)
  {
	collectionRoundStarted = false;
	queueLocked = false;
	asyncOp = nullptr;
  }
  void DispatchAsync(adal::ExternalSolverFacade& facade)
  {   
	while (!queue.empty())
	{
	  facade.DispatchMessage(*queue.front());
	  queue.pop_front();
	}
	collectionRoundStarted = false;
	{
	  std::lock_guard<std::mutex> guard(queueLockedMutex);
	  queueLocked = false;
	}    
	collectionRoundSemaphore.post();
  }

  boost::interprocess::interprocess_semaphore collectionRoundSemaphore;
  std::mutex queueLockedMutex;
  message_queue queue;  
  bool collectionRoundStarted;
  bool queueLocked;
  std::thread *asyncOp;
private:
  Pimpl(const Pimpl&);
  Pimpl& operator =(const Pimpl&) = delete;

  friend void adal::dispatch_thread(void *thisptr, void *f);
};

//namespace axis { namespace domain { namespace algorithms { 
	void adal::dispatch_thread(void *thisptr, void *f)
	{
		adal::ExternalSolverFacade::Pimpl *p = (adal::ExternalSolverFacade::Pimpl *)thisptr;
		adal::ExternalSolverFacade *facade = (adal::ExternalSolverFacade *)f;
		p->DispatchAsync(*facade);
	}
//} } }

adal::ExternalSolverFacade::ExternalSolverFacade( void )
{
  pimpl_ = new Pimpl();
}

adal::ExternalSolverFacade::~ExternalSolverFacade( void )
{
  delete pimpl_;
}

void adal::ExternalSolverFacade::StartResultCollectionRound( ada::ReducedNumericalModel& model )
{
  // try to acquire exclusive access; will block if a writing thread has not finished yet
  pimpl_->collectionRoundSemaphore.wait();  
  if (pimpl_->collectionRoundStarted)
  {
	pimpl_->collectionRoundSemaphore.post();
	throw axis::foundation::InvalidOperationException();
  }
  pimpl_->collectionRoundStarted = true;
  PrepareForCollectionRound(model);
}

void adal::ExternalSolverFacade::DispatchMessageAsync( const asmm::Message& message )
{
  std::lock_guard<std::mutex> guard(pimpl_->queueLockedMutex);
  if (pimpl_->queueLocked || !pimpl_->collectionRoundStarted)
  {
	throw axis::foundation::InvalidOperationException();
  }
  pimpl_->queue.push_back(&message.Clone());
}

void adal::ExternalSolverFacade::EndResultCollectionRound( void )
{
  {
	std::lock_guard<std::mutex> guard(pimpl_->queueLockedMutex);
	if (!pimpl_->collectionRoundStarted)
	{
	  throw axis::foundation::InvalidOperationException();
	}
	pimpl_->queueLocked = true;
  }
  if (pimpl_->asyncOp != nullptr)
  {
	if (pimpl_->asyncOp->joinable())
	{
	  pimpl_->asyncOp->detach();
	}
	delete pimpl_->asyncOp;
  }
 // pimpl_->asyncOp = new std::thread(&Pimpl::DispatchAsync, 
	//std::ref(*pimpl_), std::ref(*this));
  pimpl_->asyncOp = new std::thread(&dispatch_thread,
	  (void *)pimpl_, (void *)this);

//   if (asyncOp.joinable()) asyncOp.detach();
}

bool adal::ExternalSolverFacade::IsCollectionRoundActive( void ) const
{
  return pimpl_->collectionRoundStarted;
}

void adal::ExternalSolverFacade::FlushResultCollection( void )
{
  {
	std::lock_guard<std::mutex> guard(pimpl_->queueLockedMutex);
	if (!pimpl_->collectionRoundStarted)
	{
	  throw axis::foundation::InvalidOperationException();
	}
  }
  if (pimpl_->asyncOp->joinable()) pimpl_->asyncOp->join();
}
