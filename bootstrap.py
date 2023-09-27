from src.event import Event, EventContext, EventManager

event_on_content_request = Event('OnContentRequest')
event_on_content_received = Event('OnContentReceived')
event_on_cache_hit = Event('OnCacheHit')
event_on_cache_missed = Event('OnCacheMissed')
event_on_cache_full = Event('OnCacheFull')

def handle_clear_cache(ctx: EventContext): 
  ...
  raise NotImplementedError()

def handle_cache_cleanup(ctx: EventContext): 
  ...
  raise NotImplementedError()

def handle_cache_content(ctx: EventContext): 
  ...
  raise NotImplementedError()

def handle_clear_content_buf(ctx: EventContext): 
  ...
  raise NotImplementedError()

def handle_prune_cache(ctx: EventContext): 
  ...
  raise NotImplementedError()

def handle_request(ctx: EventContext): 
  ...
  raise NotImplementedError()

def handle_logging_shutdown(ctx: EventContext): 
  ...
  raise NotImplementedError()

def handle_dump_local_vars(ctx: EventContext): 
  ...
  raise NotImplementedError()


def dispatch_default(event_manager: EventManager):
  for event in [
    event_on_content_request,
    event_on_content_received,
    event_on_cache_hit,
    event_on_cache_missed,
    event_on_cache_full,
  ]:
    event_manager.attach_event(event)