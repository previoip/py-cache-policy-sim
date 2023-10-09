import typing as t
from src.event import Event, EventContext, EventManager
from src.pseudo_server import Server
from src.cache import T_SIZE

class EventParamContentRequest(t.NamedTuple):
  client: Server
  timestamp: int
  user_id: t.Any
  item_id: int
  item_size: T_SIZE


class EventParamContentRequestStatus(t.NamedTuple):
  client: Server
  timestamp: int
  user_id: t.Any
  item_id: int
  item_size: T_SIZE
  status: str

event_on_content_request = Event('OnContentRequest')
event_on_db_hit = Event('OnDatabaseHit')
event_on_db_missed = Event('OnDatabaseMissed')
event_on_cache_hit = Event('OnCacheHit')
event_on_cache_missed = Event('OnCacheMissed')

# ======================================================

def handle_clear_cache(ctx: EventContext, event_param): 
  ctx.event_target.cache.clear()
  return 0

# ======================================================

def handle_request(ctx: EventContext, event_param): 
  server = ctx.event_target

  if server.cache.has(event_param.item_id):
    server.cache.move_to_end(event_param.item_id)
    server.event_manager.trigger_event('OnCacheHit', 
      event_param=EventParamContentRequestStatus(*event_param, 'cache_hit')
    )
  else:
    server.event_manager.trigger_event('OnCacheMissed', 
      event_param=EventParamContentRequestStatus(*event_param, 'cache_missed')
    )
    if server.has_database():
      if server.database.has(event_param.item_id):
        server.event_manager.trigger_event('OnDatabaseHit', 
          event_param=EventParamContentRequestStatus(*event_param, 'db_hit')
        )
      else:
        server.event_manager.trigger_event('OnDatabaseMissed', 
          event_param=EventParamContentRequestStatus(*event_param, 'db_missed')
        )

    elif not server.parent is None:
      # propagate/delegate to parent server
      server.parent.event_manager.trigger_event('OnContentRequest', event_param=event_param)

  return 0

event_on_content_request.add_listener(handle_request)

# ======================================================

def handle_cache_content(ctx: EventContext, event_param): 
  ctx.event_target.cache.add(event_param.item_id, None, 1)
  return 0

event_on_content_request.add_listener(handle_cache_content)

# ======================================================

def handle_log_request(ctx: EventContext, event_param):
  if not hasattr(ctx.event_target, 'request_log_database'):
    return 1
  ctx.event_target.request_log_database.add_entry(
    event_param.timestamp,
    event_param.user_id,
    event_param.item_id
  )
  return 0

event_on_content_request.add_listener(handle_log_request)

# ======================================================

def handle_log_request_status(ctx: EventContext, event_param):
  if not hasattr(ctx.event_target, 'request_status_log_database'):
    return 1
  ctx.event_target.request_status_log_database.add_entry(
    event_param.timestamp,
    event_param.user_id,
    event_param.item_id,
    event_param.status
  )
  return 0

event_on_db_hit.add_listener(handle_log_request_status)
event_on_db_missed.add_listener(handle_log_request_status)
event_on_cache_hit.add_listener(handle_log_request_status)
event_on_cache_missed.add_listener(handle_log_request_status)

# ======================================================

# def handle_print_param(ctx: EventContext, event_param):
#   print('dequeued param:', event_param)
#   return 0

# event_on_content_request.add_listener(handle_print_param)

# ======================================================

def set_default_event(event_manager: EventManager):
  for event in [
    event_on_content_request,
    event_on_db_hit,
    event_on_db_missed,
    event_on_cache_hit,
    event_on_cache_missed,
  ]:
    event_manager.attach_event(event)