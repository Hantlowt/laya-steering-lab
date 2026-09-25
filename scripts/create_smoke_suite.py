"""Create the checked-in offline smoke suite. This fixture is human-authored, not LLM output."""

from __future__ import annotations

import hashlib
from pathlib import Path

from laya_steering.io import save_suite
from laya_steering.schemas import DecisionSchema, Example, GenerationRecord, SuiteManifest, TaskSpec

TASKS = [
    (
        "support_priority",
        "customer_support",
        ["URGENT", "NORMAL"],
        "Classify support tickets by whether immediate human intervention is required.",
        {
            "URGENT": "outage, security, or material financial harm",
            "NORMAL": "routine help without immediate harm",
        },
        [
            ("All customers receive a 503 error at checkout.", "URGENT"),
            ("An attacker changed my recovery email.", "URGENT"),
            ("We were charged twice for the annual plan.", "URGENT"),
            ("How do I update my avatar?", "NORMAL"),
            ("Please explain how to export a report.", "NORMAL"),
            ("Can I rename a workspace?", "NORMAL"),
            ("The production API started rejecting every request.", "URGENT"),
            ("Where is the dark-mode setting?", "NORMAL"),
            ("Invoices are being sent to the wrong company.", "URGENT"),
            ("I need instructions for inviting a colleague.", "NORMAL"),
            ("Our live service is unreachable from every region.", "URGENT", "sp-1"),
            ("Nobody can access the deployed application worldwide.", "URGENT", "sp-1"),
            ("A minor typo appears on the status page during a broad outage.", "URGENT"),
            ("The word urgent is in my signature; I only need profile help.", "NORMAL"),
        ],
    ),
    (
        "support_routing",
        "customer_support",
        ["BILLING", "TECHNICAL", "ACCOUNT"],
        "Route a customer request to the responsible support team.",
        {
            "BILLING": "payments and invoices",
            "TECHNICAL": "product failures",
            "ACCOUNT": "identity and access settings",
        },
        [
            ("Refund the duplicate card payment.", "BILLING"),
            ("The invoice tax number is wrong.", "BILLING"),
            ("The desktop app crashes on launch.", "TECHNICAL"),
            ("Webhooks return malformed payloads.", "TECHNICAL"),
            ("I cannot reset my password.", "ACCOUNT"),
            ("Change the owner of our workspace.", "ACCOUNT"),
            ("Why did my subscription price change?", "BILLING"),
            ("Uploads freeze at ninety percent.", "TECHNICAL"),
            ("Remove a former employee's login.", "ACCOUNT"),
            ("My receipt lists the wrong currency.", "BILLING"),
            ("The program closes whenever I attach a file.", "TECHNICAL", "sr-1"),
            ("Adding an attachment makes the client terminate.", "TECHNICAL", "sr-1"),
            ("I cannot log in to download an invoice; the password works elsewhere.", "ACCOUNT"),
            ("A billing page button is visually misaligned but charges work.", "TECHNICAL"),
        ],
    ),
    (
        "email_action",
        "email",
        ["REPLY", "ARCHIVE", "ESCALATE"],
        "Choose the operational action for an incoming business email.",
        {
            "REPLY": "needs a routine response",
            "ARCHIVE": "informational or unsolicited",
            "ESCALATE": "requires authority or urgent handling",
        },
        [
            ("Could you confirm Tuesday's meeting time?", "REPLY"),
            ("Please send the latest product sheet.", "REPLY"),
            ("Weekly industry newsletter, issue 42.", "ARCHIVE"),
            ("Automated social network digest.", "ARCHIVE"),
            ("Legal notice: response required by tomorrow.", "ESCALATE"),
            ("The regulator requests an incident report today.", "ESCALATE"),
            ("Can you clarify the final paragraph?", "REPLY"),
            ("Your monthly usage summary is ready.", "ARCHIVE"),
            ("CEO approval is required before the deadline.", "ESCALATE"),
            ("Would Thursday morning work instead?", "REPLY"),
            ("Counsel demands preservation of records by end of day.", "ESCALATE", "ea-1"),
            ("Before today ends, legal requires all records retained.", "ESCALATE", "ea-1"),
            ("A newsletter quotes a lawsuit but asks nothing of us.", "ARCHIVE"),
            ("Routine scheduling request marked HIGH IMPORTANCE by sender.", "REPLY"),
        ],
    ),
    (
        "email_sensitivity",
        "email",
        ["PUBLIC", "INTERNAL", "CONFIDENTIAL"],
        "Classify the permitted disclosure level of an email.",
        {
            "PUBLIC": "safe for public distribution",
            "INTERNAL": "company-only routine material",
            "CONFIDENTIAL": "secrets, personal data, or privileged material",
        },
        [
            ("Published press release attached.", "PUBLIC"),
            ("Link to our public launch blog.", "PUBLIC"),
            ("Team lunch is moved to noon.", "INTERNAL"),
            ("Draft agenda for the all-hands meeting.", "INTERNAL"),
            ("Customer passport scan and bank details attached.", "CONFIDENTIAL"),
            ("Attorney-client investigation notes.", "CONFIDENTIAL"),
            ("Approved public event announcement.", "PUBLIC"),
            ("Office access procedure for employees.", "INTERNAL"),
            ("Unreleased acquisition terms.", "CONFIDENTIAL"),
            ("Updated internal holiday calendar.", "INTERNAL"),
            ("Private keys for the staging service are included below.", "CONFIDENTIAL", "es-1"),
            ("Below are credentials that unlock our preproduction system.", "CONFIDENTIAL", "es-1"),
            ("Public brochure forwarded with a private customer phone number.", "CONFIDENTIAL"),
            ("Document titled secret contains only the published privacy policy.", "PUBLIC"),
        ],
    ),
    (
        "log_severity",
        "operations_logs",
        ["INFO", "WARN", "CRITICAL"],
        "Assign operational severity to a system event.",
        {
            "INFO": "normal operation",
            "WARN": "degradation needing attention",
            "CRITICAL": "active outage or data loss",
        },
        [
            ("Scheduled backup completed successfully.", "INFO"),
            ("Worker joined the pool.", "INFO"),
            ("Disk usage reached 82 percent.", "WARN"),
            ("Retry rate exceeded the warning threshold.", "WARN"),
            ("Primary database is unavailable.", "CRITICAL"),
            ("Irrecoverable corruption detected in customer records.", "CRITICAL"),
            ("Certificate expires in fourteen days.", "WARN"),
            ("Cache warmed in 120 ms.", "INFO"),
            ("All payment workers stopped responding.", "CRITICAL"),
            ("A replica is five minutes behind.", "WARN"),
            ("No writes can reach the main datastore.", "CRITICAL", "ls-1"),
            ("The authoritative database rejects every write.", "CRITICAL", "ls-1"),
            ("ERROR token appears in a successful parser test.", "INFO"),
            ("Health endpoint is green while customer writes are being lost.", "CRITICAL"),
        ],
    ),
    (
        "log_subsystem",
        "operations_logs",
        ["NETWORK", "STORAGE", "AUTH"],
        "Route an infrastructure event to the owning subsystem.",
        {
            "NETWORK": "connectivity and DNS",
            "STORAGE": "disks, databases, and persistence",
            "AUTH": "credentials and authorization",
        },
        [
            ("DNS lookup timed out for the upstream host.", "NETWORK"),
            ("Packet loss crossed ten percent.", "NETWORK"),
            ("Volume has no free blocks.", "STORAGE"),
            ("Database checkpoint failed to flush.", "STORAGE"),
            ("OAuth token signature is invalid.", "AUTH"),
            ("Role policy denied the request.", "AUTH"),
            ("TLS connection reset at the gateway.", "NETWORK"),
            ("Object store returned checksum mismatch.", "STORAGE"),
            ("Session credential expired.", "AUTH"),
            ("Resolver returned NXDOMAIN unexpectedly.", "NETWORK"),
            ("The authorization service rejected a valid bearer token.", "AUTH", "lu-1"),
            ("A legitimate access token was refused by identity validation.", "AUTH", "lu-1"),
            ("Database connection failed because DNS cannot resolve its host.", "NETWORK"),
            ("Login audit file cannot be written because the disk is full.", "STORAGE"),
        ],
    ),
    (
        "return_eligibility",
        "ecommerce",
        ["ELIGIBLE", "INELIGIBLE", "REVIEW"],
        "Decide whether a merchandise return is allowed under policy.",
        {
            "ELIGIBLE": "within policy",
            "INELIGIBLE": "clearly outside policy",
            "REVIEW": "exception or insufficient evidence",
        },
        [
            ("Unopened item received five days ago with receipt.", "ELIGIBLE"),
            ("Wrong size returned within the thirty-day window.", "ELIGIBLE"),
            ("Consumable was fully used before the return request.", "INELIGIBLE"),
            ("Purchase was made eighteen months ago.", "INELIGIBLE"),
            ("Gift has no receipt but may be in the extended holiday window.", "REVIEW"),
            ("Damaged item has conflicting delivery dates.", "REVIEW"),
            ("Sealed accessory delivered last week.", "ELIGIBLE"),
            ("Personalized engraving matches the approved proof.", "INELIGIBLE"),
            ("Carrier loss claim and return request overlap.", "REVIEW"),
            ("Opened hygiene product with no defect.", "INELIGIBLE"),
            (
                "The unused product arrived nine days ago and proof of purchase is present.",
                "ELIGIBLE",
                "re-1",
            ),
            (
                "Receipt included; item is untouched and less than two weeks old.",
                "ELIGIBLE",
                "re-1",
            ),
            ("Return is one day late because the buyer was hospitalized.", "REVIEW"),
            ("Box is open, but the policy explicitly permits inspection.", "ELIGIBLE"),
        ],
    ),
    (
        "order_risk",
        "ecommerce",
        ["ALLOW", "REVIEW", "BLOCK"],
        "Assess fraud risk for an online order.",
        {
            "ALLOW": "ordinary low-risk purchase",
            "REVIEW": "suspicious but inconclusive",
            "BLOCK": "strong evidence of abuse",
        },
        [
            ("Returning customer ships to their usual address.", "ALLOW"),
            ("Small order uses a previously verified card.", "ALLOW"),
            ("New account places an unusually large overnight order.", "REVIEW"),
            ("Billing country differs from IP location.", "REVIEW"),
            ("Card is confirmed stolen and used across twenty accounts.", "BLOCK"),
            ("Known chargeback ring fingerprint matched.", "BLOCK"),
            ("First purchase includes three high-value gift cards.", "REVIEW"),
            ("Normal reorder from a corporate account.", "ALLOW"),
            ("Device belongs to a banned fraud cluster.", "BLOCK"),
            ("Traveling customer uses verified 2FA abroad.", "ALLOW"),
            ("A compromised payment token appears on many synthetic identities.", "BLOCK", "or-1"),
            ("Many fabricated accounts share one stolen credential.", "BLOCK", "or-1"),
            ("Huge order is unusual but was preapproved by the account manager.", "ALLOW"),
            ("The note says fraud test, but all verified signals are normal.", "ALLOW"),
        ],
    ),
    (
        "home_event",
        "smart_home",
        ["NOTIFY", "AUTOMATE", "IGNORE"],
        "Choose how a smart-home controller should handle an event.",
        {
            "NOTIFY": "human awareness is needed",
            "AUTOMATE": "safe predefined action",
            "IGNORE": "benign noise",
        },
        [
            ("Smoke detector reports sustained smoke.", "NOTIFY"),
            ("Front door opens while everyone is away.", "NOTIFY"),
            ("Sunset occurs and occupied-room lights are off.", "AUTOMATE"),
            ("Temperature falls below the heating setpoint.", "AUTOMATE"),
            ("Motion sensor sends its scheduled heartbeat.", "IGNORE"),
            ("Battery reading repeats unchanged.", "IGNORE"),
            ("Water leak sensor activates under the sink.", "NOTIFY"),
            ("Morning schedule requests blinds to open.", "AUTOMATE"),
            ("Duplicate presence event arrives within one second.", "IGNORE"),
            ("Carbon monoxide rises above the safe level.", "NOTIFY"),
            ("A leak probe detects water beside the boiler.", "NOTIFY", "he-1"),
            ("The boiler-room moisture sensor reports active flooding.", "NOTIFY", "he-1"),
            ("A test-mode smoke alarm is clearly labeled as a drill.", "IGNORE"),
            ("Routine heartbeat arrives with an actual low-battery alert.", "NOTIFY"),
        ],
    ),
    (
        "energy_action",
        "smart_home",
        ["REDUCE", "SHIFT", "NO_ACTION"],
        "Choose an energy-management response to a household condition.",
        {
            "REDUCE": "lower consumption now",
            "SHIFT": "defer flexible work",
            "NO_ACTION": "leave operation unchanged",
        },
        [
            ("Grid emergency begins while optional heating is high.", "REDUCE"),
            ("Peak tariff starts with decorative lights on.", "REDUCE"),
            ("Dishwasher is queued and cheap power begins in two hours.", "SHIFT"),
            ("EV charging can finish during tonight's low tariff.", "SHIFT"),
            ("Solar surplus exceeds current household demand.", "NO_ACTION"),
            ("Critical medical device is drawing normal power.", "NO_ACTION"),
            ("Air conditioner cools an empty room during a grid alert.", "REDUCE"),
            ("Laundry cycle can wait until off-peak hours.", "SHIFT"),
            ("Baseline usage is already below the target.", "NO_ACTION"),
            ("Flexible water heating overlaps the price peak.", "SHIFT"),
            ("Delay the nonurgent vehicle charge until rates drop.", "SHIFT", "en-1"),
            ("Postpone flexible EV charging to the cheaper period.", "SHIFT", "en-1"),
            ("High price period starts, but life-support equipment must continue.", "NO_ACTION"),
            ("A message says reduce, yet net usage is negative due to solar export.", "NO_ACTION"),
        ],
    ),
]


def main() -> None:
    tasks, examples = [], []
    prompt_hash = hashlib.sha256(b"human-authored offline smoke fixture v1").hexdigest()
    generation = GenerationRecord(
        role="manual_fixture", provider="human", model="none", seed=17, prompt_sha256=prompt_hash
    )
    for name, domain, labels, description, class_descriptions, rows in TASKS:
        task = TaskSpec(
            name=name,
            domain=domain,
            description=description,
            decision=DecisionSchema(
                labels=labels, class_descriptions=class_descriptions, question=description
            ),
            policy=description,
        )
        tasks.append(task)
        split_sizes = (
            ("specialization", 6),
            ("validation", 2),
            ("hidden", 2),
            ("paraphrase", 2),
            ("hard", 2),
        )
        cursor = 0
        for split, count in split_sizes:
            for offset, value in enumerate(rows[cursor : cursor + count]):
                text, label, *pair = value
                examples.append(
                    Example(
                        id=f"{name}-{split[:4]}-{offset:03d}",
                        input=text,
                        label=label,
                        split=split,
                        task_name=name,
                        domain=domain,
                        pair_id=pair[0] if pair else None,
                        source_style="human-smoke-v1",
                        tags=["boundary"] if split == "hard" else [],
                    )
                )
            cursor += count
    manifest = SuiteManifest(
        name="offline-smoke-v1",
        tasks=[x.name for x in tasks],
        seed=17,
        specialization_generation=generation,
        benchmark_generation=generation,
        metadata={
            "scientific_evidence": False,
            "note": "Human-authored CI fixture; use LLM-generated suites for research.",
        },
    )
    save_suite(Path("benchmarks/smoke"), manifest, tasks, examples)


if __name__ == "__main__":
    main()
