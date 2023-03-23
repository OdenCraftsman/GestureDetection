# GestureDetection

## 注意

本リポジトリは、実行を想定されていないため、ローカルリポジトリを作成しても動作しません。

動作させるためには、hands up（挙手）、 hands wave（腕振り）、 others（その他ジェスチャ）の動画をそれぞれ一時間程度撮影する必要があります。

参考程度に閲覧していただけると幸いです。

## フォルダ説明
- setup/data/origin

    このフォルダには、該当するカテゴリフォルダにもととなる動画データを格納します。

- setup/tester

    このフォルダでは、完成したモデルをテストするための動画データを格納します。

- model/gesture

    このフォルダにはmodelup.pyによって作成されたモデルが格納されます。

- model/openpose

    このフォルダには、骨格推定用のモデルを格納します。以下のサイトからbody_pose_model.pthをダウンロードし、格納してください。
    
    [骨格推定モデル](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0)