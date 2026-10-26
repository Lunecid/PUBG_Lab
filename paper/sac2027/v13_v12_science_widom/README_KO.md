# SAC 2027 MLA v13 편집본

v12의 과학적 내용에 Widom 버전의 설명 구조를 적용한 9쪽 영문 원고다. 원본 v12는 수정하지 않았다. 변경 내역과 검증 범위는 `CHANGES_KO.md`에 있다.

## ZIP 구성

- `main.tex`: 편집 가능한 전체 원고
- `refs.bib`: v12의 원본 참고문헌 파일
- `main.bbl`: 검증된 BibTeX 출력
- `figures/*.pdf`: 원본 벡터 그림 3개
- `acmart.cls`, `ACM-Reference-Format.bst`: 공식 SAC 2027 배포 템플릿
- `TEMPLATE_README.txt`: 템플릿의 원본 안내·라이선스
- `SAC2027_MLA_v13_2026-10-08.pdf`: 제공 PDF
- `CHANGES_KO.md`, `README_KO.md`: 변경 설명과 사용 안내

## Overleaf

ZIP을 새 프로젝트로 가져오고 main document를 `main.tex`, compiler를 **XeLaTeX**로 선택한다. 참고문헌은 BibTeX로 처리된다. 로컬 경로나 프로젝트 밖의 파일에 의존하지 않는다.

## 로컬 컴파일

Tectonic으로 다음 명령을 실행한다.

```sh
tectonic --keep-logs --keep-intermediates main.tex
```

처음 실행할 때 필요한 TeX 패키지와 글꼴이 다운로드될 수 있다. 결과는 `main.pdf`다. 이 버전은 Tectonic으로 컴파일 및 PDF 표시를 검증했다.

TeX Live 환경에서는 다음과 같이 빌드할 수 있다.

```sh
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Overleaf나 별도 TeX Live 버전의 실제 빌드까지 검증한 것은 아니다. 엔진·패키지 버전에 따라 줄바꿈이 조금 달라질 수 있다.

## 투고 전

원고는 익명 검토용이다. 저자 정보와 임의의 DOI·ISBN·저작권 문구를 넣지 않았다. 현재 9쪽으로 기본 8쪽보다 1쪽 길다. 최종 트랙 업로드 지침·추가 페이지 조건과 AI 활용 고지의 모델·버전 및 저자 검토 진술을 확인한다. 새 실험은 수행하지 않았다.
