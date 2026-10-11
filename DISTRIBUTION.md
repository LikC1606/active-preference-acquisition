# Complete offline submission

Download the two attachments from the complete-submission release. Both are required.

| Attachment | Bytes | SHA-256 |
| --- | ---: | --- |
| active-preference-acquisition.tar.gz.001 | 1900000000 | b7e1a14a5d08df41916169f7f79ad020c7bfae595eff4e1d1fdc65d8d7a6cba5 |
| active-preference-acquisition.tar.gz.002 | 1791439871 | 4c76086853280ea84643bcf433860b56eb18f00fffcad7ad0fc2f290577f59f9 |

```sh
cat active-preference-acquisition.tar.gz.001 active-preference-acquisition.tar.gz.002 > active-preference-acquisition.tar.gz
sha256sum active-preference-acquisition.tar.gz
mkdir delivery-restored
tar -xzf active-preference-acquisition.tar.gz -C delivery-restored
```

Combined SHA-256: `db2f9d65ab8722eb29f11a9f29a549e947157399e893b6a00914aa43c41c7531`.

The archive is 3,691,439,871 bytes and restores 4,008 files in workspace/, expert_evidence/ and optimization_evidence/. All original hashes and modes passed. Splitting only addresses GitHub's attachment limit; it does not reduce the total data size. No training or model API requests were made. Formal platform acceptance remains pending.
