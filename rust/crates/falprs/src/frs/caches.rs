//! FRS caches: stream group tokens, common and default configs, video stream configs, face
//! descriptors, faces attached to video streams, special groups and their faces.
//!
//! Rows marked as deleted are dropped from the caches by incremental updates and removed from the
//! database later by a maintenance task, as in the C++ service.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use pg_cluster::{Cluster, HostType};
use serde_json::Value;

use super::config::{CommonConfig, VStreamConfig};

/// Stream group id by access token.
pub type GroupsData = HashMap<String, i32>;

pub async fn load_groups(cluster: Cluster, host_type: HostType) -> Result<GroupsData> {
    let rows = cluster
        .query(host_type, "select auth_token::varchar, id_group from vstream_groups", &[])
        .await?;
    Ok(rows.iter().filter_map(|row| Some((row.get(0), row.get::<_, Option<i32>>(1)?))).collect())
}

/// Common configs and default video stream configs by stream group id.
#[derive(Debug, Default)]
pub struct ConfigData {
    pub common: HashMap<i32, CommonConfig>,
    pub default_vstream: HashMap<i32, VStreamConfig>,
}

pub async fn load_config(cluster: Cluster, host_type: HostType) -> Result<ConfigData> {
    let rows = cluster
        .query(
            host_type,
            r#"
            select
              vg.id_group,
              coalesce(cc.config, '{}') || coalesce(dvc.config, '{}') config
            from
              vstream_groups vg
              left join common_config cc
                on cc.id_group = vg.id_group
              left join default_vstream_config dvc
                on dvc.id_group = vg.id_group
            "#,
            &[],
        )
        .await?;
    let mut data = ConfigData::default();
    for row in &rows {
        let id_group: i32 = row.get(0);
        let config: Value = row.get(1);
        data.common.insert(id_group, CommonConfig::from_json(&config));
        let mut default_vstream = VStreamConfig::from_json(&config);
        default_vstream.id_group = id_group;
        data.default_vstream.insert(id_group, default_vstream);
    }
    Ok(data)
}

/// Video stream configs by `<id_group>_<stream ID>`.
pub type VStreamsConfigData = HashMap<String, Arc<VStreamConfig>>;

pub fn vstream_key(id_group: i32, vstream_ext: &str) -> String {
    format!("{id_group}_{vstream_ext}")
}

pub struct VStreamConfigRow {
    key: String,
    config: Option<VStreamConfig>,
}

pub async fn load_vstreams_config(
    cluster: Cluster,
    host_type: HostType,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<VStreamConfigRow>> {
    const QUERY: &str = r#"
        select
          concat(vs.id_group, '_', vs.vstream_ext) unique_key,
          vs.id_group,
          vs.id_vstream,
          vs.vstream_ext,
          coalesce(vs.url, '') url,
          coalesce(vs.callback_url, '') callback_url,
          coalesce(vs.callback_url_barcodes, '') callback_url_barcodes,
          coalesce(d.config, '{}') || coalesce(vs.config, '{}') config,
          vs.flag_deleted
        from
          video_streams vs
          left join default_vstream_config d
            on d.id_group = vs.id_group
    "#;
    let rows = match since {
        None => cluster.query(host_type, QUERY, &[]).await?,
        Some(since) => {
            let query = format!("{QUERY} where last_updated >= $1");
            cluster.query(host_type, &query, &[&since]).await?
        }
    };
    Ok(rows
        .iter()
        .map(|row| {
            let flag_deleted: bool = row.get(8);
            let config = (!flag_deleted).then(|| {
                let json: Value = row.get(7);
                let mut config = VStreamConfig::from_json(&json);
                // A video stream without a stream group is unreachable through the API
                config.id_group = row.get::<_, Option<i32>>(1).unwrap_or_default();
                config.id_vstream = row.get(2);
                config.vstream_ext = row.get(3);
                config.url = row.get(4);
                config.callback_url = row.get(5);
                config.callback_url_barcodes = row.get(6);
                config
            });
            VStreamConfigRow { key: row.get(0), config }
        })
        .collect())
}

pub fn apply_vstreams_config(data: &mut VStreamsConfigData, rows: Vec<VStreamConfigRow>) {
    for row in rows {
        match row.config {
            Some(config) => {
                data.insert(row.key, Arc::new(config));
            }
            None => {
                data.remove(&row.key);
            }
        }
    }
}

/// A face descriptor normalized to unit length.
pub type Descriptor = Arc<[f32]>;

/// Reads a descriptor stored as raw native-endian floats.
pub fn descriptor_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub fn descriptor_to_bytes(descriptor: &[f32]) -> Vec<u8> {
    descriptor.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Scales a descriptor to unit L2 norm (a zero descriptor is left as is), as `fd / cv::norm(fd)`.
pub fn normalize(descriptor: &mut [f32]) {
    let norm = descriptor.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
    let norm = if norm <= 0.0 { 1.0 } else { norm };
    for v in descriptor {
        *v = (f64::from(*v) / norm) as f32;
    }
}

#[derive(Debug, Default, Clone)]
pub struct FaceDescriptorData {
    pub descriptors: HashMap<i32, Descriptor>,
    /// Parent descriptor id by spawned descriptor id.
    pub spawned_to_parent: HashMap<i32, i32>,
}

pub struct FaceDescriptorRow {
    id_descriptor: i32,
    descriptor: Option<Descriptor>,
    id_parent: i32,
}

pub async fn load_face_descriptors(
    cluster: Cluster,
    host_type: HostType,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<FaceDescriptorRow>> {
    const QUERY: &str =
        "select id_descriptor, descriptor_data, flag_deleted, coalesce(id_parent, 0) id_parent from face_descriptors";
    let rows = match since {
        None => cluster.query(host_type, QUERY, &[]).await?,
        Some(since) => {
            let query = format!("{QUERY} where last_updated >= $1");
            cluster.query(host_type, &query, &[&since]).await?
        }
    };
    Ok(rows
        .iter()
        .map(|row| {
            let flag_deleted: bool = row.get(2);
            let descriptor = (!flag_deleted).then(|| {
                let bytes: Option<&[u8]> = row.get(1);
                let mut descriptor = descriptor_from_bytes(bytes.unwrap_or_default());
                normalize(&mut descriptor);
                Descriptor::from(descriptor)
            });
            FaceDescriptorRow {
                id_descriptor: row.get(0),
                descriptor,
                id_parent: row.get(3),
            }
        })
        .collect())
}

pub fn apply_face_descriptors(data: &mut FaceDescriptorData, rows: Vec<FaceDescriptorRow>) {
    for row in rows {
        match row.descriptor {
            Some(descriptor) => {
                data.descriptors.insert(row.id_descriptor, descriptor);
                if row.id_parent > 0 {
                    data.spawned_to_parent.insert(row.id_descriptor, row.id_parent);
                }
            }
            None => {
                data.descriptors.remove(&row.id_descriptor);
                data.spawned_to_parent.remove(&row.id_descriptor);
            }
        }
    }
}

/// Descriptor ids (attached ones and the descriptors spawned from them) by video stream id.
pub type VStreamDescriptorsData = HashMap<i32, HashSet<i32>>;

pub struct VStreamDescriptorRow {
    id_vstream: i32,
    id_descriptor: i32,
    id_spawned: i32,
    flag_deleted: bool,
}

pub async fn load_vstream_descriptors(
    cluster: Cluster,
    host_type: HostType,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<VStreamDescriptorRow>> {
    const QUERY: &str = r#"
        select
          ldv.id_vstream,
          ldv.id_descriptor,
          coalesce(fd.id_descriptor, 0) id_spawned,
          ldv.flag_deleted
        from
          link_descriptor_vstream ldv
          left join face_descriptors fd
            on ldv.id_descriptor = fd.id_parent
    "#;
    let rows = match since {
        None => cluster.query(host_type, QUERY, &[]).await?,
        Some(since) => {
            let query = format!("{QUERY} where ldv.last_updated >= $1");
            cluster.query(host_type, &query, &[&since]).await?
        }
    };
    Ok(rows
        .iter()
        // Both link columns are nullable in the schema
        .filter_map(|row| {
            Some(VStreamDescriptorRow {
                id_vstream: row.get::<_, Option<i32>>(0)?,
                id_descriptor: row.get::<_, Option<i32>>(1)?,
                id_spawned: row.get(2),
                flag_deleted: row.get(3),
            })
        })
        .collect())
}

pub fn apply_vstream_descriptors(data: &mut VStreamDescriptorsData, rows: Vec<VStreamDescriptorRow>) {
    for row in rows {
        if row.flag_deleted {
            if let Some(descriptors) = data.get_mut(&row.id_vstream) {
                descriptors.remove(&row.id_descriptor);
                if row.id_spawned > 0 {
                    descriptors.remove(&row.id_spawned);
                }
            }
        } else {
            let descriptors = data.entry(row.id_vstream).or_default();
            descriptors.insert(row.id_descriptor);
            if row.id_spawned > 0 {
                descriptors.insert(row.id_spawned);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SGConfig {
    pub id_special_group: i32,
    pub callback_url: String,
    pub id_group: i32,
}

/// Special groups, including the ones marked as deleted, as in the C++ service.
#[derive(Debug, Default)]
pub struct SGConfigData {
    /// Special groups by API token.
    pub by_token: HashMap<String, SGConfig>,
    /// API token by special group id.
    pub token_by_id: HashMap<i32, String>,
    /// Special group ids by stream group id.
    pub by_group: HashMap<i32, HashSet<i32>>,
}

impl SGConfigData {
    pub fn by_id(&self, id_special_group: i32) -> Option<&SGConfig> {
        self.by_token.get(self.token_by_id.get(&id_special_group)?)
    }
}

pub async fn load_sg_config(cluster: Cluster, host_type: HostType) -> Result<SGConfigData> {
    let rows = cluster
        .query(
            host_type,
            "select sg_api_token, id_special_group, coalesce(callback_url, '') callback_url, id_group from special_groups",
            &[],
        )
        .await?;
    let mut data = SGConfigData::default();
    for row in &rows {
        let token: String = row.get(0);
        let config = SGConfig {
            id_special_group: row.get(1),
            callback_url: row.get(2),
            id_group: row.get::<_, Option<i32>>(3).unwrap_or_default(),
        };
        data.token_by_id.insert(config.id_special_group, token.clone());
        data.by_group.entry(config.id_group).or_default().insert(config.id_special_group);
        data.by_token.insert(token, config);
    }
    Ok(data)
}

/// Descriptor ids by special group id.
pub type SGDescriptorsData = HashMap<i32, HashSet<i32>>;

pub struct SGDescriptorRow {
    id_sgroup: i32,
    id_descriptor: i32,
    flag_deleted: bool,
}

pub async fn load_sg_descriptors(
    cluster: Cluster,
    host_type: HostType,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<SGDescriptorRow>> {
    const QUERY: &str = "select id_sgroup, id_descriptor, flag_deleted from link_descriptor_sgroup";
    let rows = match since {
        None => cluster.query(host_type, QUERY, &[]).await?,
        Some(since) => {
            let query = format!("{QUERY} where last_updated >= $1");
            cluster.query(host_type, &query, &[&since]).await?
        }
    };
    Ok(rows
        .iter()
        .map(|row| SGDescriptorRow {
            id_sgroup: row.get(0),
            id_descriptor: row.get(1),
            flag_deleted: row.get(2),
        })
        .collect())
}

pub fn apply_sg_descriptors(data: &mut SGDescriptorsData, rows: Vec<SGDescriptorRow>) {
    for row in rows {
        if row.flag_deleted {
            if let Some(descriptors) = data.get_mut(&row.id_sgroup) {
                descriptors.remove(&row.id_descriptor);
            }
        } else {
            data.entry(row.id_sgroup).or_default().insert(row.id_descriptor);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_round_trip_and_normalization() {
        let bytes = descriptor_to_bytes(&[3.0, 4.0]);
        let mut descriptor = descriptor_from_bytes(&bytes);
        assert_eq!(descriptor, vec![3.0, 4.0]);
        normalize(&mut descriptor);
        assert_eq!(descriptor, vec![0.6, 0.8]);
        let mut zero = vec![0.0, 0.0];
        normalize(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0]);
    }

    #[test]
    fn vstream_descriptors_follow_marks() {
        let mut data = VStreamDescriptorsData::default();
        let row = |id_descriptor, id_spawned, flag_deleted| VStreamDescriptorRow {
            id_vstream: 1,
            id_descriptor,
            id_spawned,
            flag_deleted,
        };
        apply_vstream_descriptors(&mut data, vec![row(10, 0, false), row(11, 12, false)]);
        assert_eq!(data[&1], HashSet::from([10, 11, 12]));
        apply_vstream_descriptors(&mut data, vec![row(11, 12, true)]);
        assert_eq!(data[&1], HashSet::from([10]));
    }
}
