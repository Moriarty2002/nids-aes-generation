"""Registered synthetic offline-PCAP dataset for independently evaluated raw actions.

Uses the explicitly scoped IPv4 prefix extractor for BOTH clean training and
post-mutation re-extraction. The first-20 occupancy and float64 measurements
are authoritative for this declared *offline study only*; this is not an
unverified replacement for CICIoMT2024's deferred raw-PCAP extractor.
Experiment identity binds every raw capture's SHA-256 plus its ordered label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from constraints.rules import validate_clean_packet_source
from core.packet_schema import CICIOMT2024_FIELDS, CICIOMT2024_PAD_VALUES
from core.provenance import _file_sha256
from core.registry import register_dataset
from preprocessing.offline_packet_pcap import OfflinePcapPacketPrefix
from .ciciomt2024 import SequenceParquetDataset


@register_dataset('offline_pcap_study')
class OfflinePcapStudyDataset(SequenceParquetDataset):
    """Manifest with bounded immutable PCAP samples and declared numeric 0/1 labels."""
    def __init__(self, path: str | Path, *, attacker_ip: str, flatten: bool = False,
                 max_length: int = 20):
        if max_length != 20:
            raise ValueError('NIDS sequence policy is the first 20 real packets')
        self.path = Path(path)
        manifest = json.loads(self.path.read_text(encoding='utf-8'))
        rows = manifest.get('samples')
        if not isinstance(rows, list) or len(rows) < 3:
            raise ValueError('Offline PCAP manifest needs at least 3 labeled flow records')
        self.sample_ids = tuple(str(row['id']) for row in rows)
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError('Offline PCAP identities must be unique')
        self._labels = np.asarray([row['label'] for row in rows], dtype=np.int64)
        if set(np.unique(self._labels)) != {0, 1}:
            raise ValueError('Offline PCAP benchmark needs explicit numeric benign 0 and malicious 1')
        files = [self.path.parent / str(row['pcap']) for row in rows]
        if not all(p.is_file() and p.suffix.lower() == '.pcap' for p in files):
            raise ValueError('Every registered offline PCAP must exist as classic .pcap')
        content_digest = hashlib.sha256()
        for identity, file in zip(self.sample_ids, files, strict=True):
            content_digest.update(identity.encode()); content_digest.update(bytes.fromhex(_file_sha256(file)))
        self._capture_sha256 = content_digest.hexdigest()
        self.raw_traffic = files
        self.fields = CICIOMT2024_FIELDS
        self.max_length = 20
        self.sequence_policy = 'first_20_real_packets_prefix_v1'
        self.flatten = bool(flatten)
        self.include_padding_indicator = False
        self.padding_indicator_name = 'FEAT_PAD_MASK'
        self.pad_values = dict(CICIOMT2024_PAD_VALUES)
        self.strict_feat_pad = True
        self.truncate = True
        self.feat_pad_column = 'FEAT_PAD'
        self.label_column = 'label'
        self.benign_string = '0'
        self.task_mode = 'binary'
        self.label_mode = 'binary'
        self.classes = ('0', '1')
        self.class_mapping = {'0':0, '1':1}
        self.id_column = 'id'
        self.group_column = None
        self.normalization = None
        self.normalization_path = None
        self.group_ids = tuple(str(row['group']) for row in rows) if all('group' in row for row in rows) else None
        self._extractor = OfflinePcapPacketPrefix(attacker_ip=attacker_ip)
        parsed = self._extractor.transform(files)
        self._canonical = parsed.physical_features
        self.valid_mask = parsed.valid_mask
        validate_clean_packet_source(self._canonical, self.fields, self.valid_mask,
                                     self.pad_values, self.sample_ids)
        self._canonical_shape = self._canonical.shape
        self.lengths = self.valid_mask.sum(axis=1).astype(np.int64)
        self.truncated_flow_count = int(sum(v>20 for v in parsed.metadata['raw_packet_counts']))
        self.discarded_packet_positions = int(sum(max(v-20,0) for v in parsed.metadata['raw_packet_counts']))
        self.original_real_packet_counts_summary = {}
        self._raw_cache = None
        self._normalized_cache = None

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state['clean_physical_validation_policy'] = ('offline_ipv4_packet_prefix_v1:' + self._capture_sha256)
        return state
