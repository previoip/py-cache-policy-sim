import typing as t
from src.event import Event, EventContext, EventManager
from src.pseudo_server import Server
from src.cache import T_SIZE

class EventParamContentRequest(t.NamedTuple):
  client: Server
  user_id: t.Any
  item_id: int
  item_size: T_SIZE

event_on_content_request = Event('OnContentRequest')
event_on_content_received = Event('OnContentReceived')
event_on_db_hit = Event('OnDatabaseHit')
event_on_db_missed = Event('OnDatabaseMissed')
event_on_cache_hit = Event('OnCacheHit')
event_on_cache_missed = Event('OnCacheMissed')
event_on_cache_full = Event('OnCacheFull')

event_on_cache_update = Event('OnCacheUpdate')

# ======================================================

def handle_clear_cache(ctx: EventContext, event_param): 
  ctx.event_target.cache.evict()
  return 0

event_on_cache_update.add_listener(handle_clear_cache)

# ======================================================

def handle_request(ctx: EventContext, event_param): 
  ...

  ctx.event_manager.trigger_event('OnCacheUpdate')
  return 0

# ======================================================

def handle_logging_shutdown(ctx: EventContext, event_param): 
  ...
  raise NotImplementedError()

# ======================================================

def handle_log_request(ctx: EventContext, event_param):
  if not hasattr(ctx.event_target, 'request_log_database'):
    return 1
  ctx.event_target.request_log_database.add_entry(
    ctx.event_target.timer(),
    event_param.user_id,
    event_param.item_id
  )
  return 0

event_on_content_request.add_listener(handle_log_request)


# ======================================================

def handle_cache_content(ctx: EventContext, event_param): 
  ctx.event_target.cache.add(event_param.item_id, None, 1)
  return 0

event_on_content_request.add_listener(handle_cache_content)

# ======================================================

def handle_print_param(ctx: EventContext, event_param):
  print('unqueued param:', event_param)
  return 0

event_on_content_request.add_listener(handle_print_param)

# ======================================================

def handle_print_context(ctx: EventContext, event_param):
  print('invoked context:', ctx)
  return 0

event_on_cache_update.add_listener(handle_print_context)

# ======================================================

def set_default_event(event_manager: EventManager):
  for event in [
    event_on_content_request,
    event_on_content_received,
    event_on_cache_hit,
    event_on_cache_missed,
    event_on_cache_full,
    event_on_cache_update,
  ]:
    event_manager.attach_event(event)