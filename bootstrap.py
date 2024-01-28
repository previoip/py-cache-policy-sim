import typing as t
from src.event import Event, EventContext, EventManager
from src.pseudo_server import Server
from src.cache import T_SIZE

class ItemStatus:
  cache_hit = 'cache_hit'
  cache_miss = 'cache_miss'
  db_hit = 'cache_hit'
  db_miss = 'cache_miss'

class EventParamContentRequest(t.NamedTuple):
  request_id: int
  client: Server
  timestamp: int
  user_id: t.Any
  item_id: int
  rating: t.Union[int, float]
  item_size: T_SIZE


class EventParamContentRequestStatus(t.NamedTuple):
  request_id: int
  client: Server
  timestamp: int
  user_id: t.Any
  item_id: int
  rating: t.Union[int, float]
  item_size: T_SIZE
  status: str

event_on_content_request = Event('OnContentRequest')
event_on_db_hit = Event('OnDatabaseHit')
event_on_db_missed = Event('OnDatabaseMissed')
event_on_cache_hit = Event('OnCacheHit')
event_on_cache_missed = Event('OnCacheMissed')

subroutine_cache = Event('SubCache')

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
      event_param=EventParamContentRequestStatus(*event_param, ItemStatus.cache_hit)
    )
  else:
    server.event_manager.trigger_event('OnCacheMissed', 
      event_param=EventParamContentRequestStatus(*event_param, ItemStatus.cache_miss)
    )
    if server.has_database():
      if server.database.has(event_param.item_id):
        server.event_manager.trigger_event('OnDatabaseHit', 
          event_param=EventParamContentRequestStatus(*event_param, ItemStatus.db_hit)
        )
      else:
        server.event_manager.trigger_event('OnDatabaseMissed', 
          event_param=EventParamContentRequestStatus(*event_param, ItemStatus.db_miss)
        )

    elif not server.is_root():
      # propagate/delegate to parent server
      server.parent.event_manager.trigger_event('OnContentRequest', event_param=event_param)

  if not server.cfg.flag_suppress_cache_on_req:
    server.event_manager.trigger_event('SubCache', event_param=event_param)

  ctx.event_target.states.request_counter += 1
  return 0

event_on_content_request.add_listener(handle_request)

# ======================================================

def handle_cache_content(ctx: EventContext, event_param): 
  ctx.event_target.cache.add(event_param.item_id, event_param.rating, event_param.item_size)
  return 0

subroutine_cache.add_listener(handle_cache_content)

# ======================================================

def handle_cache_eviction_policy(ctx: EventContext, event_param): 
  ctx.event_target.cache.evict()
  return 0

subroutine_cache.add_listener(handle_cache_eviction_policy)

# ======================================================

def handle_log_request(ctx: EventContext, event_param):
  if not hasattr(ctx.event_target, 'request_log_database'):
    return 1
  ctx.event_target.request_log_database.add_entry(
    event_param.request_id,
    event_param.timestamp,
    event_param.user_id,
    event_param.item_id,
    event_param.rating
  )
  return 0

event_on_content_request.add_listener(handle_log_request)

# ======================================================

def handle_log_request_status(ctx: EventContext, event_param):
  if not hasattr(ctx.event_target, 'request_status_log_database'):
    return 1
  ctx.event_target.request_status_log_database.add_entry(
    event_param.request_id,
    event_param.timestamp,
    event_param.user_id,
    event_param.item_id,
    event_param.rating,
    event_param.status
  )
  return 0

event_on_db_hit.add_listener(handle_log_request_status)
event_on_db_missed.add_listener(handle_log_request_status)
event_on_cache_hit.add_listener(handle_log_request_status)
event_on_cache_missed.add_listener(handle_log_request_status)

# ======================================================

def handle_incr_states(ctx: EventContext, event_param):
  if event_param.status == ItemStatus.cache_hit:
    ctx.event_target.states.cache_hit_counter += 1
  elif event_param.status == ItemStatus.cache_miss:
    ctx.event_target.states.cache_miss_counter += 1
  return 0

event_on_cache_hit.add_listener(handle_incr_states)
event_on_cache_missed.add_listener(handle_incr_states)

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
    subroutine_cache,
  ]:
    event_manager.attach_event(event)