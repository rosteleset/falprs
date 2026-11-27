-- Migrate explicit ViT dimensions. A stream can inherit its model name from its group.
do $$
declare
    dimension text;
begin
    foreach dimension in array array['lpc-net-input-width', 'lpc-net-input-height'] loop
        update default_vstream_config
        set config = jsonb_set(config, array[dimension], '256')
        where config->>dimension = '224'
          and coalesce(config->>'lpc-net-model-name', 'lpcnet_vit') = 'lpcnet_vit';

        update vstreams s
        set config = jsonb_set(s.config, array[dimension], '256'), last_updated = now()
        where s.config->>dimension = '224'
          and coalesce(
              s.config->>'lpc-net-model-name',
              (select d.config->>'lpc-net-model-name' from default_vstream_config d where d.id_group = s.id_group),
              'lpcnet_vit'
          ) = 'lpcnet_vit';
    end loop;
end;
$$;
