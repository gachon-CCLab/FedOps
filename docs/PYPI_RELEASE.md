# Git · PyPI release alignment

FedOps 메인 라이브러리는 **X 서버 `/home/ccl/Desktop/forPYPI`에서만 게시한다.**
기존 `pypi_fedops` 환경과 Twine 설정을 재사용하며 Mac/GitHub에서 직접 게시하지 않는다.

## 자동 배포 흐름

1. `main` 반영 → GitHub `FedOps PyPI release`가 OS별 host(macOS/Windows/Linux), client, server evaluation, model contract 테스트와 wheel 설치 검사를 실행한다.
2. X 서버가 주기적으로 `origin/main`을 확인하고 **정확히 같은 commit의 main push CI 성공**을 확인한다. 실패·진행 중·조회 오류는 게시하지 않는다.
3. 격리된 worktree에서 테스트를 다시 실행하고 wheel/sdist를 빌드한다. 기존 개발 checkout은 변경하지 않는다.
4. 기존 X 서버 Twine으로 게시한 후 PyPI의 source fingerprint·파일 SHA-256·설치된 CLI를 확인한다. 모두 통과해야 `last-successful-revision`을 갱신한다.

Git 반영과 PyPI 게시는 원자적이지 않다. 배포 지연/실패 중에는 일치 완료로 보고하지 않는다.
테스트 통과는 모든 Task의 무결함 보장이 아니며 주요 변경은 실제 Studio/FL 시나리오 테스트도 필요하다.

## 버전 규칙

- 패키지 코드/README/metadata 변경은 `src/python/setup.py` 버전을 올린다.
- 이미 게시된 버전은 source fingerprint와 wheel 내용이 일치할 때만 게시를 생략한다. 다르면 실패: 새 버전 필요.
- 기존 버전을 삭제하거나 덮어쓰지 않는다. 기존 Registry Task의 immutable Runtime pin도 자동 변경하지 않는다.
- CI에서 생성하는 `fedops/_release.json`은 Git revision과 source fingerprint를 담으며 직접 편집/커밋하지 않는다.

## X 서버 운영 경로

| 용도 | 경로 |
|---|---|
| 공식 배포 루트 | `/home/ccl/Desktop/forPYPI` |
| 기존 Git 저장소 | `FedOps_github/FedOps` |
| 자동 게시 스크립트 | `automation/x_server_publish.sh` |
| 회귀 테스트용 격리 환경 | `release-test-env` |
| 게시 artifact 보관 | `releases/<version>/<commit>/` |
| 마지막 검증 성공 | `automation/last-successful-revision` |

기존 `/home/ccl/anaconda3/envs/pypi_fedops` Twine과 `/home/ccl/.pypirc`를 사용한다.
인증값은 열람·출력·복사하지 않는다. Git도 기존 remote/credential helper만 비대화형으로 사용한다.

수동 재시도도 X 서버에서 동일 스크립트를 실행한다.

```bash
bash /home/ccl/Desktop/forPYPI/automation/x_server_publish.sh
```

인증 실패는 기존 서버의 게시 인증을 관리자가 복구한 뒤 재시도한다. 토큰을 채팅이나 Git에 저장하지 않는다.
일부 artifact만 게시된 불완전 릴리스는 성공으로 처리하지 않는다. `main` CI 실패는 소스를 수정해야 하며 우회 게시하지 않는다.
