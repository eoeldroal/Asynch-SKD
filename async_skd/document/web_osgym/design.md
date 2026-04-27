# Web / OS Gym Integration Design

## 0. 목적

이 문서는 `verl`의 agent-loop 기반 rollout에 **WebGym / OSWorld 계열 remote environment protocol**을 통합할 때의 상위 설계 원칙을 정리한다.

여기서 말하는 environment는 다음 특성을 가진다.

- 외부 서버가 세션을 관리한다
- observation은 screenshot과 a11y tree를 포함한다
- action은 `computer_13` 계열 low-level action schema를 따른다
- 최종 reward는 environment가 계산해 응답한다

이 문서의 목표는 "어떤 상태 기계와 책임 분리 위에 이 기능을 올릴 것인가"를 고정하는 것이다.  
구체적인 클래스 이름, 함수 시그니처, 내부 helper 구성은 이후 구현 단계에서 바뀔 수 있다.

## 1. 설계 목표

통합의 목표는 다음과 같다.

1. `verl`의 기존 agent-loop 구조를 최대한 유지한다.
2. environment 세션 시작, action 적용, reward 회수를 명확히 분리한다.
3. screenshot / a11y tree를 모델 입력으로 자연스럽게 제공한다.
4. `DONE` / `FAIL`과 시스템 종료를 모두 일관된 종료 프로토콜로 수렴시킨다.
5. environment가 계산한 최종 reward를 현재 강화학습 업데이트 경로에 자연스럽게 실어 나른다.

## 2. 비목표

이 문서는 다음을 다루지 않는다.

- low-level HTTP client 구현 상세
- 구체적인 prompt 문구 또는 serializer 포맷
- tokenizer, image preprocessing, transport retry 정책의 세부 코드
- 특정 모델 전용 tool-call 파서의 세부 구현
- reward shaping이나 dense intermediate reward 설계

즉 이 문서는 **구현 문서가 아니라 구조 문서**다.

## 3. 핵심 관점

이 integration은 일반적인 계산 tool을 하나 더 붙이는 작업이 아니다.  
본질적으로는 **stateful remote environment를 tool-agent loop 안에 넣는 작업**이다.

따라서 핵심은 다음 두 가지다.

- environment는 trajectory 동안 유지되는 **세션**을 가진다
- tool response는 단순 실행 결과가 아니라 **새 observation** 역할을 한다

즉 이 통합에서 tool은 "코드를 실행해 문자열을 돌려주는 함수"가 아니라,  
**정책이 상호작용하는 외부 환경의 step function**에 가깝다.

## 4. 프로토콜 추상화

환경 서버는 세 종류의 요청을 가진다.

### 4.1 Start

세션을 열고 최초 observation을 가져온다.

- 입력: `task_id`, `request_id`, 기타 시작 옵션
- 출력: 최초 screenshot, a11y tree

### 4.2 Action

현재 세션에 action을 적용하고 다음 observation을 가져온다.

- 입력: action list
- 출력: 다음 screenshot, a11y tree, 또는 action failure message

### 4.3 Reward

종료 후 최종 reward를 가져온다.

- 입력: 세션 식별자
- 출력: scalar reward

이 세 요청은 environment protocol 차원에서는 분리되지만, agent-loop 차원에서는 하나의 trajectory lifecycle로 묶여야 한다.

## 5. 상태 기계 매핑

현재 `ToolAgentLoop` / `SkdAgentLoop`의 큰 상태 구조를 유지하면서 environment protocol을 매핑하는 것이 기본 방향이다.

### 5.1 `PENDING`

`PENDING`은 environment 세션을 시작하고 최초 observation을 확보하는 단계다.

여기서 수행되는 논리는 다음과 같다.

- `task_id`를 바탕으로 environment에 `start` 요청을 보낸다
- 최초 screenshot과 a11y tree를 받는다
- 이를 task instruction과 함께 초기 모델 입력 메시지로 구성한다

즉 `PENDING`은 더 이상 단순 prompt 준비 단계가 아니라,  
**환경과의 첫 동기화 단계**가 된다.

### 5.2 `GENERATING`

모델이 다음 행동을 제안하는 단계다.

모델은 화면과 a11y tree를 보고 다음 행동을 tool-call 형태로 생성한다.  
이 출력은 내부적으로 environment action schema로 정규화될 수 있어야 한다.

중요한 점은, 이 단계의 책임은 어디까지나 **행동 제안**이지 환경 적용이 아니라는 것이다.

### 5.3 `PROCESSING_TOOLS`

모델이 제안한 action을 실제 environment에 적용하는 단계다.

여기서 수행되는 논리는 다음과 같다.

- 모델 출력에서 action을 파싱한다
- 이를 environment `action` 요청으로 보낸다
- 새 screenshot, a11y tree, failure message를 받는다
- 이를 다음 턴의 모델 입력으로 사용할 observation으로 만든다

이 단계에서 tool response는 사실상 **environment observation carrier**다.

### 5.4 `TERMINATED`

environment interaction이 끝난 상태다.  
이후에는 추가 action을 생성하지 않으며, 최종 reward가 회수되어 있어야 한다.

## 6. 초기 observation 원칙

모델은 첫 턴부터 다음 세 가지를 함께 받아야 한다.

- task instruction
- 현재 screenshot
- 현재 a11y tree

즉 environment integration의 첫 관문은 "모델이 action을 낼 준비가 되었는가"이지,  
"서버 연결이 되었는가"만이 아니다.

이 원칙 때문에 최초 `start` 응답은 단순 side-effect가 아니라,  
**실질적인 initial observation**으로 취급되어야 한다.

## 7. Action 표현 원칙

environment action은 `computer_13` 계열 low-level action schema를 따른다.

여기서 중요한 설계 원칙은 다음과 같다.

- 모델 관점에서는 action space가 단일해야 한다
- `CLICK`, `SCROLL`, `TYPE`, `WAIT`, `DONE`, `FAIL`은 모두 같은 표면 action schema 안에 존재할 수 있다
- 내부 구현은 필요할 경우 이를 일반 action과 종료 action으로 나누어 해석할 수 있다

즉 **표면 schema는 통일하되, 상태 기계에서의 의미는 다를 수 있다**는 것이 기본 원칙이다.

## 8. 종료 의미론

종료는 두 갈래에서 발생한다.

### 8.1 모델 주도 종료

모델이 `DONE` 또는 `FAIL`을 action으로 생성하는 경우다.

이 경우 `DONE` / `FAIL`은 겉으로는 action schema 안에 있지만,  
실질적으로는 **종료 요청 action**이다.

따라서 이 경우 loop는:

1. environment에 종료 의도를 전달하고
2. 최종 reward를 요청하고
3. 현재 trajectory를 종료해야 한다

### 8.2 시스템 주도 종료

예:

- `max_length`
- `max_chunks`
- 기타 시스템 hard stop

이 경우 모델이 종료 action을 명시적으로 내리지는 않았지만,  
loop는 일관성을 위해 동일한 종료 수순을 따라야 한다.

즉:

1. 더 이상의 generation/action 반복을 중단하고
2. environment에 대해 reward를 요청하고
3. trajectory를 종료한다

## 9. 종료 경로 통일 원칙

종료 이유와 관계없이 마지막은 하나의 공통 규칙으로 수렴하는 것이 좋다.

**종료 시에는 항상 최종 reward를 한 번 회수하고 loop를 끝낸다.**

구체적으로는:

- `DONE` / `FAIL`: 종료 action 이후 reward 회수
- `max_length` / `max_chunks`: 별도 종료 action 없이 reward 회수

즉 종료 이유는 다를 수 있지만, 학습 시스템이 받는 마지막 산출물은 항상 **final scalar reward**여야 한다.

## 10. Reward 통합 원칙

이 integration에서 reward는 외부 reward model이 아니라 **environment가 계산한 최종 answer**다.

따라서 reward의 source of truth는 environment server다.

중요한 설계 원칙은:

- reward는 agent loop 종료 시점에 회수한다
- trainer가 environment protocol을 직접 알 필요는 없다
- agent loop는 회수한 최종 reward를 기존 RL batch 경로에 실어 나른다

즉 reward 통합은 "trainer가 서버를 직접 호출한다"가 아니라,  
**agent loop가 environment reward를 받아 trajectory output의 일부로 넘긴다**는 구조가 더 자연스럽다.

## 11. Tool response의 의미

이 통합에서 tool response는 일반적인 "실행 결과 문자열"보다 넓은 의미를 가진다.

실질적으로 tool response는 다음 중 하나를 담는다.

- 새 screenshot
- 새 a11y tree
- action failure message
- 종료 이후의 reward 관련 상태

즉 이 tool은 일반 utility tool이 아니라,  
**모델에게 다음 observation을 공급하는 environment bridge**다.

## 12. Session lifecycle 원칙

이 integration은 trajectory 동안 environment session을 유지해야 한다.

따라서 다음 원칙이 필요하다.

- 세션 시작은 한 trajectory당 한 번이어야 한다
- action은 같은 세션 위에서 순차적으로 누적되어야 한다
- 종료 후에는 reward를 회수하고 세션을 닫아야 한다

즉 이 tool은 stateless one-shot tool이 아니라,  
**trajectory-lifetime session tool**이어야 한다.

## 13. Failure 처리 원칙

environment action 실패는 transport-level exception과 동일하게 다루면 안 된다.

action failure 응답은 정책에게 중요한 observation일 수 있다.  
예를 들어 focus가 안 맞아서 typing이 실패했다는 정보는 다음 action을 바꾸는 데 직접 필요하다.

따라서 기본 원칙은:

- action failure는 가능하면 **관측 가능한 환경 응답**으로 모델에 다시 보여준다
- 단순히 예외를 던지고 trajectory를 끊는 방향은 기본값이 아니다

즉 실패도 environment state의 일부로 취급한다.

## 14. Validation / Training 해석

이 integration은 원칙적으로 training rollout뿐 아니라 validation rollout에도 적용될 수 있다.  
다만 validation은 teacher guidance를 평가하는 단계가 아니라 **student policy 자체를 평가하는 단계**로 해석해야 한다.

즉 environment integration 자체와 async SKD의 teacher-guided semantics를 섞어 생각하지 않는 것이 중요하다.

## 15. 안정적인 추상 기준

이 문서에서 바뀌지 말아야 하는 기준은 다음과 같다.

1. `PENDING`은 environment `start`와 initial observation 획득 단계다
2. `GENERATING`은 행동 제안 단계다
3. `PROCESSING_TOOLS`는 environment action 적용과 다음 observation 획득 단계다
4. `DONE` / `FAIL`은 표면상 action schema에 있으나 의미상 종료 요청이다
5. 시스템 종료와 모델 종료는 모두 최종 reward 회수로 수렴한다
6. reward의 source of truth는 environment server다
7. environment session은 trajectory 동안 유지되어야 한다
8. tool response는 observation carrier로 이해해야 한다

## 16. 한 줄 요약

이 통합을 가장 짧게 표현하면 다음과 같다.

**Web / OS Gym integration은 `tool_agent` 위에 stateful remote environment session을 얹고, start/action/reward를 agent-loop 상태 기계에 대응시켜 최종 environment reward를 기존 RL 업데이트 경로로 연결하는 설계다.**
