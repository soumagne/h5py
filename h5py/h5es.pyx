"""
   H5ES API low-level Cython bindings.
"""

include "config.pxi"

cdef class EventStackID(ObjectID):
   """
   Represents an HDF5 event stack identifier
   """
   pass


def create():
   """() => EventStackID

   Create an event stack.
   """

   return EventStackID.open(H5EScreate())


# def cancel(EventStackID es not None, int event_idx not None):
#    """(EventStackID es, INT event_idx) => INT status
# 
#    Cancels the in-progress event specified by event_idx in the event stack
#    specified by es and returns the event’s status.  
#    """
# 
#    cdef H5ES_status_t status
# 
#    H5EScancel(es.id, event_idx, &status)
#    return status
#
# 
# def cancel_all(EventStackID es not None):
#    """(EventStackID es)
# 
#    Cancels all in-progress events in the event stack specified by es and
#    indicates their statuses.
#    """
# 
#    cdef H5ES_status_t *status
#    H5EScancel_all(es.id, &status)
#    return status


def clear(EventStackID es not None):
   """(EventStackID es)

   Clear all events from an event stack specified by es after first confirming
   that no in-progress events remain in the stack. If the stack does contain
   one or more in-progress events, no events are cleared and the call fails (a
   negative value is returned).
   """

   H5ESclear(es.id)


def close(EventStackID es not None):
   """(EventStackID es)

   Closes an event stack specified by es.
   """

   H5ESclose(es.id)


def get_count(EventStackID es):
   """(EventStackID es) => INT count

   Returns the number of events in the event stack specified by es.
   """

   cdef size_t count
   H5ESget_count(es.id, &count)
   return count


def test(EventStackID es not None, int event_idx not None):
   """(EventStackID es) => INT status

   Reports the completion status of the event specified by event_idx in the
   event stack specified by es.
   """

   cdef H5ES_status_t status
   H5EStest(es.id, event_idx, &status)
   return status


def test_all(EventStackID es):
   """(EventStackID es) => INT status

   Reports an overall completion status for all events in the event stack
   specified by es.
   """

   cdef H5ES_status_t status
   H5EStest_all(es.id, &status)
   return status


def wait(EventStackID es not None, int event_idx not None):
   """(EventStackID es, INT event_idx) => INT status

   Waits for the completion or cancellation of the event specified by
   event_idx in the event stack specified by es and reports the event’s
   completion status.  The call does not return until the event being waited
   on completes or is cancelled.
   """

   cdef H5ES_status_t status
   H5ESwait(es.id, event_idx, &status)
   return status


def wait_all(EventStackID es not None):
   """(EventStackID es) => INT status

   Waits for the completion or cancellation of all events in the event stack
   specified by es and reports an overall completion status.  The call does
   not return until all events in the event stack complete or are cancelled.
   """

   cdef H5ES_status_t status
   H5ESwait_all(es.id, &status)
   return status
