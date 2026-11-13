-- The service talks to Triton over gRPC (port 8001). Installations that used the Triton HTTP
-- client have the HTTP port 8000 in the inference server addresses of every kind of stored config;
-- it is replaced with 8001 wherever it is set. Other ports are deliberate and are left as they are.
update
  video_streams
set
  config = (
    select jsonb_object_agg(
        key,
        case
            when key like '%-inference-server' and jsonb_typeof(value) = 'string'
                then to_jsonb(regexp_replace(value #>> '{}', ':8000$', ':8001'))
            else value
        end
    )
    from jsonb_each(config)
  ),
  last_updated = now()
where
  case
      when jsonb_typeof(config) = 'object'
          then exists (select 1 from jsonb_each_text(config) where key like '%-inference-server' and value ~ ':8000$')
      else false
  end;

update
  default_vstream_config
set
  config = (
    select jsonb_object_agg(
        key,
        case
            when key like '%-inference-server' and jsonb_typeof(value) = 'string'
                then to_jsonb(regexp_replace(value #>> '{}', ':8000$', ':8001'))
            else value
        end
    )
    from jsonb_each(config)
  )
where
  case
      when jsonb_typeof(config) = 'object'
          then exists (select 1 from jsonb_each_text(config) where key like '%-inference-server' and value ~ ':8000$')
      else false
  end;

update
  common_config
set
  config = (
    select jsonb_object_agg(
        key,
        case
            when key like '%-inference-server' and jsonb_typeof(value) = 'string'
                then to_jsonb(regexp_replace(value #>> '{}', ':8000$', ':8001'))
            else value
        end
    )
    from jsonb_each(config)
  )
where
  case
      when jsonb_typeof(config) = 'object'
          then exists (select 1 from jsonb_each_text(config) where key like '%-inference-server' and value ~ ':8000$')
      else false
  end;
