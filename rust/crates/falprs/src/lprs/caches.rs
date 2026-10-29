//! LPRS caches: stream group tokens, video stream configs and video stream membership.

use std::collections::HashMap;

use anyhow::Result;
use pg_cluster::{Cluster, HostType};
use serde_json::Value;

use super::vstream_config::VStreamConfig;

/// Stream group id by access token.
pub type GroupsData = HashMap<String, i32>;

pub async fn load_groups(cluster: Cluster, host_type: HostType) -> Result<GroupsData> {
    let rows = cluster
        .query(host_type, "select auth_token::varchar, id_group from vstream_groups", &[])
        .await?;
    Ok(rows.iter().filter_map(|row| Some((row.get(0), row.get::<_, Option<i32>>(1)?))).collect())
}

/// A video stream with its effective config: its stream group's default config overlaid with its own.
#[derive(Debug, Clone)]
pub struct VStreamEntry {
    pub id_vstream: i32,
    pub ext_id: String,
    pub config: VStreamConfig,
}

/// Video stream configs by `<id_group>_<stream ID>`; a stream group's default config is under `<id_group>`.
pub type VStreamsConfigData = HashMap<String, VStreamEntry>;

pub fn vstream_key(id_group: i32, ext_id: &str) -> String {
    format!("{id_group}_{ext_id}")
}

const SQL_VSTREAMS_CONFIG: &str = r#"
    select
      concat(vs.id_group, '_', vs.ext_id) unique_key,
      vs.id_group,
      vs.id_vstream,
      vs.ext_id,
      coalesce(d.config, '{}') || coalesce(vs.config, '{}') config
    from
      vstreams vs
      left join default_vstream_config d
        on d.id_group = vs.id_group

    union all

    select
      d.id_group::text unique_key,
      d.id_group,
      0 id_vstream,
      'v' ext_id,
      d.config
    from
      default_vstream_config d
"#;

pub async fn load_vstreams_config(cluster: Cluster, host_type: HostType) -> Result<VStreamsConfigData> {
    let rows = cluster.query(host_type, SQL_VSTREAMS_CONFIG, &[]).await?;
    // vstreams.id_group is nullable; a video stream without a stream group is unreachable through the API
    Ok(rows
        .iter()
        .filter_map(|row| {
            row.get::<_, Option<i32>>(1)?;
            let config = row.get::<_, Option<Value>>(4).unwrap_or(Value::Null);
            let entry = VStreamEntry {
                id_vstream: row.get(2),
                ext_id: row.get(3),
                config: VStreamConfig::from_json(&config),
            };
            Some((row.get(0), entry))
        })
        .collect())
}

/// Stream group id by video stream id.
pub type VStreamGroupData = HashMap<i32, i32>;

pub async fn load_vstream_group(cluster: Cluster, host_type: HostType) -> Result<VStreamGroupData> {
    let rows = cluster
        .query(host_type, "select id_vstream, id_group from vstreams", &[])
        .await?;
    Ok(rows.iter().filter_map(|row| Some((row.get(0), row.get::<_, Option<i32>>(1)?))).collect())
}
